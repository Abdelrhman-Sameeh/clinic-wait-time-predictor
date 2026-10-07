"""Offline tests for clinic-information retrieval and answering."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from app.db import save_clinic_config
from app.rag import (
    MAX_DISTANCE,
    answer_question,
    build_index,
    build_knowledge_document,
    build_knowledge_sections,
    chunk_sections,
    rebuild_index_from_db,
    retrieve,
)
from app.scheduler import ConfigNotFoundError
from app.schemas import ClinicConfig, QuestionRequest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CONFIG_PATH = PROJECT_ROOT / "data" / "sample_clinic_config.json"
EXPECTED_SECTIONS = [
    "Clinic Information",
    "Working Days and Hours",
    "Breaks",
    "Appointments and Booking Rules",
    "Peak Hours and Waiting Times",
    "Late Arrival Policy",
    "Cancellation Policy",
    "Walk-in Policy",
    "Emergency Policy",
    "No-show Policy",
    "Services and Prices",
    "Special Conditions",
]


class HashedBagOfWords:
    """Deterministic small embedding function for offline Chroma tests."""

    def name(self) -> str:
        """Return Chroma's stable embedding-function identifier."""
        return "test-hashed-bag-of-words"

    def __call__(self, input: list[str]) -> list[list[float]]:
        vectors = []
        for text in input:
            vector = [0.0] * 2048
            for word in re.findall(r"[a-z0-9]+", text.casefold()):
                position = int.from_bytes(
                    hashlib.blake2b(word.encode(), digest_size=4).digest(),
                    "big",
                ) % len(vector)
                vector[position] += 1.0
            vectors.append(vector)
        return vectors

    def embed_query(self, input: list[str]) -> list[list[float]]:
        """Embed query text using the same deterministic representation."""
        return self(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        """Embed stored documents using the same deterministic representation."""
        return self(input)


@pytest.fixture
def rag_setup(tmp_path: Path) -> tuple[ClinicConfig, str, str, HashedBagOfWords]:
    """Prepare an isolated clinic database and vector collection."""
    config = ClinicConfig.model_validate_json(
        SAMPLE_CONFIG_PATH.read_text(encoding="utf-8")
    )
    db_path = str(tmp_path / "clinic.sqlite")
    persist_dir = str(tmp_path / "chroma")
    embedding_function = HashedBagOfWords()
    save_clinic_config(config, path=db_path)
    return config, db_path, persist_dir, embedding_function


def test_knowledge_sections_cover_sample_config(
    rag_setup: tuple[ClinicConfig, str, str, HashedBagOfWords],
) -> None:
    config, _, _, _ = rag_setup
    sections = build_knowledge_sections(config)

    assert list(sections) == EXPECTED_SECTIONS
    assert config.phone in sections["Clinic Information"]
    assert "15 minutes" in sections["Appointments and Booking Rules"]
    assert config.late_arrival_policy in sections["Late Arrival Policy"]
    assert all(service.name in sections["Services and Prices"] for service in config.services)


def test_knowledge_document_formats_times_as_twelve_hour_clock(
    rag_setup: tuple[ClinicConfig, str, str, HashedBagOfWords],
) -> None:
    config, _, _, _ = rag_setup
    document = build_knowledge_document(config)

    assert "4:00 PM" in document
    assert "6:30 PM to 6:45 PM" in document
    assert "16:00:00" not in document


def test_empty_optional_lists_produce_readable_sections(
    rag_setup: tuple[ClinicConfig, str, str, HashedBagOfWords],
) -> None:
    config, _, _, _ = rag_setup
    config = config.model_copy(
        update={"breaks": [], "services": [], "special_conditions": None}
    )
    sections = build_knowledge_sections(config)

    assert "No scheduled breaks" in sections["Breaks"]
    assert "No services or prices" in sections["Services and Prices"]
    assert "Special Conditions" not in sections


def test_chunk_sections_splits_oversized_text_and_keeps_section_name() -> None:
    section = "Clinic Information"
    long_text = " ".join(
        f"Sentence number {index} describes clinic information." for index in range(60)
    )

    chunks = chunk_sections({section: long_text})

    assert len(chunks) > 1
    assert all(set(chunk) == {"id", "section", "text"} for chunk in chunks)
    assert all(chunk["section"] == section for chunk in chunks)
    assert all(section in chunk["text"] for chunk in chunks)
    assert len({chunk["id"] for chunk in chunks}) == len(chunks)


def test_build_index_replaces_existing_collection_without_duplicates(
    rag_setup: tuple[ClinicConfig, str, str, HashedBagOfWords],
) -> None:
    config, _, persist_dir, embedding_function = rag_setup
    expected_count = len(chunk_sections(build_knowledge_sections(config)))

    assert build_index(config, persist_dir, embedding_function) == expected_count
    assert build_index(config, persist_dir, embedding_function) == expected_count
    results = retrieve(
        "late arrival policy",
        persist_dir=persist_dir,
        embedding_function=embedding_function,
        k=20,
    )
    assert len(results) == expected_count


def test_retrieve_finds_late_arrival_section(
    rag_setup: tuple[ClinicConfig, str, str, HashedBagOfWords],
) -> None:
    config, _, persist_dir, embedding_function = rag_setup
    build_index(config, persist_dir, embedding_function)

    results = retrieve(
        "late arrival grace minutes policy",
        persist_dir=persist_dir,
        embedding_function=embedding_function,
    )

    assert results[0]["section"] == "Late Arrival Policy"
    assert "distance" in results[0]


def test_answer_question_returns_only_real_source_sections(
    rag_setup: tuple[ClinicConfig, str, str, HashedBagOfWords],
) -> None:
    config, db_path, persist_dir, embedding_function = rag_setup
    build_index(config, persist_dir, embedding_function)
    response = {
        "answer": "The late arrival policy is available.",
        "source_sections": ["Late Arrival Policy", "Invented Section"],
        "found_in_clinic_info": True,
    }
    calls: list[str] = []

    def fake_llm(system_prompt: str, user_prompt: str) -> str:
        calls.append(system_prompt)
        return json.dumps(response)

    answer = answer_question(
        QuestionRequest(question="What is the late arrival policy?"),
        db_path=db_path,
        persist_dir=persist_dir,
        embedding_function=embedding_function,
        llm=fake_llm,
    )

    assert answer.source_sections == ["Late Arrival Policy"]
    assert answer.found_in_clinic_info
    assert calls


def test_invalid_json_retries_once_then_returns_safe_fallback(
    rag_setup: tuple[ClinicConfig, str, str, HashedBagOfWords],
) -> None:
    _, db_path, persist_dir, embedding_function = rag_setup
    build_index(
        ClinicConfig.model_validate_json(
            SAMPLE_CONFIG_PATH.read_text(encoding="utf-8")
        ),
        persist_dir,
        embedding_function,
    )
    calls = 0

    def invalid_llm(system_prompt: str, user_prompt: str) -> str:
        nonlocal calls
        calls += 1
        return "not json"

    answer = answer_question(
        QuestionRequest(question="What is the late arrival policy?"),
        db_path=db_path,
        persist_dir=persist_dir,
        embedding_function=embedding_function,
        llm=invalid_llm,
    )

    assert calls == 2
    assert not answer.found_in_clinic_info
    assert answer.source_sections == []
    assert "Please contact the clinic directly" in answer.answer


def test_invalid_json_once_then_valid_response_succeeds(
    rag_setup: tuple[ClinicConfig, str, str, HashedBagOfWords],
) -> None:
    config, db_path, persist_dir, embedding_function = rag_setup
    build_index(config, persist_dir, embedding_function)
    responses = iter(
        [
            "```json\ninvalid\n```",
            json.dumps(
                {
                    "answer": "The clinic lists a late-arrival policy.",
                    "source_sections": ["Late Arrival Policy"],
                    "found_in_clinic_info": True,
                }
            ),
        ]
    )
    calls = 0

    def retrying_llm(system_prompt: str, user_prompt: str) -> str:
        nonlocal calls
        calls += 1
        return next(responses)

    answer = answer_question(
        QuestionRequest(question="What is the late arrival policy?"),
        db_path=db_path,
        persist_dir=persist_dir,
        embedding_function=embedding_function,
        llm=retrying_llm,
    )

    assert calls == 2
    assert answer.answer == "The clinic lists a late-arrival policy."
    assert answer.found_in_clinic_info


def test_unrelated_question_uses_fallback_without_calling_llm(
    rag_setup: tuple[ClinicConfig, str, str, HashedBagOfWords],
) -> None:
    config, db_path, persist_dir, embedding_function = rag_setup
    build_index(config, persist_dir, embedding_function)
    calls = 0

    def forbidden_llm(system_prompt: str, user_prompt: str) -> str:
        nonlocal calls
        calls += 1
        raise AssertionError("LLM should not be called for an uncovered question")

    answer = answer_question(
        QuestionRequest(question="What is the quantum recipe for starfish?"),
        db_path=db_path,
        persist_dir=persist_dir,
        embedding_function=embedding_function,
        llm=forbidden_llm,
    )

    assert calls == 0
    assert not answer.found_in_clinic_info
    assert config.phone in answer.answer


def test_rebuild_index_requires_saved_config(tmp_path: Path) -> None:
    with pytest.raises(ConfigNotFoundError):
        rebuild_index_from_db(
            db_path=str(tmp_path / "empty.sqlite"),
            persist_dir=str(tmp_path / "chroma"),
            embedding_function=HashedBagOfWords(),
        )


def test_question_request_rejects_two_character_question() -> None:
    with pytest.raises(ValidationError):
        QuestionRequest(question="hi")


def test_out_of_scope_response_when_all_results_exceed_max_distance(
    rag_setup: tuple[ClinicConfig, str, str, HashedBagOfWords],
) -> None:
    config, db_path, persist_dir, embedding_function = rag_setup
    build_index(config, persist_dir, embedding_function)

    results = retrieve(
        "quantum recipe starfish",
        persist_dir=persist_dir,
        embedding_function=embedding_function,
    )
    assert all(item["distance"] > MAX_DISTANCE for item in results)

    answer = answer_question(
        QuestionRequest(question="What is the quantum recipe for starfish?"),
        db_path=db_path,
        persist_dir=persist_dir,
        embedding_function=embedding_function,
        llm=lambda system, user: pytest.fail("LLM must not be called"),
    )
    assert answer.found_in_clinic_info is False
