"""Offline tests for Qwen model caching and chat-template generation."""

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from app import qwen


def test_qwen_model_loader_is_cached_and_uses_cpu_safe_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    calls: list[tuple[str, dict[str, Any]]] = []

    class FakeTokenizerLoader:
        @staticmethod
        def from_pretrained(name: str) -> object:
            calls.append(("tokenizer", {"name": name}))
            return object()

    class FakeModelLoader:
        @staticmethod
        def from_pretrained(name: str, **kwargs: Any) -> object:
            calls.append(("model", {"name": name, **kwargs}))
            class FakeModel:
                device = None
                is_eval = False

                def to(self, device: Any) -> "FakeModel":
                    self.device = device
                    return self

                def eval(self) -> "FakeModel":
                    self.is_eval = True
                    return self

            model = FakeModel()
            calls[-1][1]["result"] = model
            return model

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=FakeTokenizerLoader,
            AutoModelForCausalLM=FakeModelLoader,
        ),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(qwen, "_MODEL_LOADED", False)
    monkeypatch.setattr(qwen, "_MODEL_DEVICE", None)
    monkeypatch.setattr(qwen, "_MODEL_DTYPE", None)
    qwen._load_model.clear()
    try:
        first = qwen._load_model()
        second = qwen._load_model()
    finally:
        qwen._load_model.clear()

    assert first is second
    assert len(calls) == 2
    assert calls[0][1]["name"] == qwen.MODEL_NAME
    assert calls[1][1]["name"] == qwen.MODEL_NAME
    assert calls[1][1]["torch_dtype"] is torch.float32
    assert calls[1][1]["low_cpu_mem_usage"] is True
    assert "device_map" not in calls[1][1]
    assert first[1].device == torch.device("cpu")
    assert first[1].is_eval
    assert qwen.get_model_runtime_info() == {
        "device": "cpu",
        "dtype": "float32",
    }


@pytest.mark.parametrize(
    ("free_memory", "expected_device", "expected_dtype"),
    [
        (qwen.MIN_FREE_VRAM_BYTES, "cuda:0", "float16"),
        (qwen.MIN_FREE_VRAM_BYTES - 1, "cpu", "float32"),
    ],
)
def test_qwen_selects_cuda_only_with_at_least_four_gib_free(
    monkeypatch: pytest.MonkeyPatch,
    free_memory: int,
    expected_device: str,
    expected_dtype: str,
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda device=0: (free_memory, free_memory * 2),
    )

    device, dtype = qwen._select_device_and_dtype()

    assert str(device) == expected_device
    assert str(dtype).removeprefix("torch.") == expected_dtype


def test_qwen_answer_uses_tokenizer_chat_template_and_greedy_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    captured: dict[str, Any] = {}

    class FakeTokenizer:
        eos_token_id = 2

        def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
            captured["messages"] = messages
            captured["template_kwargs"] = kwargs
            return "rendered-chat-prompt"

        def __call__(self, prompt: str, return_tensors: str) -> dict[str, Any]:
            captured["prompt"] = prompt
            return {"input_ids": torch.tensor([[1, 2]])}

        def decode(self, tokens: Any, skip_special_tokens: bool) -> str:
            captured["decoded_tokens"] = tokens.tolist()
            return " grounded answer "

    class FakeModel:
        device = torch.device("cpu")

        def generate(self, **kwargs: Any) -> Any:
            captured["generation_kwargs"] = kwargs
            captured["inference_mode"] = torch.is_inference_mode_enabled()
            return torch.tensor([[1, 2, 3, 4]])

    monkeypatch.setattr(
        qwen,
        "_load_model",
        lambda: (FakeTokenizer(), FakeModel()),
    )

    answer = qwen.generate_answer("Use context only.", "Question and context.")

    assert answer == "grounded answer"
    assert captured["prompt"] == "rendered-chat-prompt"
    assert captured["template_kwargs"] == {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    assert captured["messages"] == [
        {"role": "system", "content": "Use context only."},
        {"role": "user", "content": "Question and context."},
    ]
    assert captured["generation_kwargs"]["max_new_tokens"] == qwen.MAX_NEW_TOKENS
    assert captured["generation_kwargs"]["do_sample"] is False
    assert captured["generation_kwargs"]["pad_token_id"] == 2
    assert captured["inference_mode"] is True
