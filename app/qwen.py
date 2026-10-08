"""Cached local Qwen generation for the clinic RAG pipeline."""

import logging
from typing import Any

import streamlit as st


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 192
MIN_FREE_VRAM_BYTES = 4 * 1024**3
_MODEL_LOADED = False
_MODEL_DEVICE: str | None = None
_MODEL_DTYPE: str | None = None
LOGGER = logging.getLogger(__name__)


def _select_device_and_dtype() -> tuple[Any, Any]:
    """Use CUDA only when at least 4 GiB of VRAM is currently free."""
    import torch

    if torch.cuda.is_available():
        try:
            free_memory, _ = torch.cuda.mem_get_info(device=0)
        except RuntimeError as exc:
            LOGGER.warning(
                "Could not inspect free CUDA memory; using CPU (%s)",
                type(exc).__name__,
            )
        else:
            if free_memory >= MIN_FREE_VRAM_BYTES:
                return torch.device("cuda:0"), torch.float16
    return torch.device("cpu"), torch.float32


def get_model_runtime_info() -> dict[str, str]:
    """Report the selected or loaded model device and dtype without loading it."""
    if _MODEL_LOADED and _MODEL_DEVICE is not None and _MODEL_DTYPE is not None:
        return {"device": _MODEL_DEVICE, "dtype": _MODEL_DTYPE}
    try:
        device, dtype = _select_device_and_dtype()
    except ImportError:
        return {"device": "unavailable", "dtype": "unavailable"}
    return {"device": str(device), "dtype": str(dtype).removeprefix("torch.")}


@st.cache_resource(show_spinner="Loading the local clinic assistant…")
def _load_model() -> tuple[Any, Any]:
    """Load the tokenizer and model once per process."""
    global _MODEL_DEVICE, _MODEL_DTYPE, _MODEL_LOADED
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device, dtype = _select_device_and_dtype()
    dtype_name = str(dtype).removeprefix("torch.")
    LOGGER.info("Loading %s on %s with %s", MODEL_NAME, device, dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    _MODEL_DEVICE = str(device)
    _MODEL_DTYPE = dtype_name
    _MODEL_LOADED = True
    return tokenizer, model


def generate_answer(system_prompt: str, user_prompt: str) -> str:
    """Generate one chat-template-formatted answer from the existing RAG prompt."""
    import torch

    tokenizer, model = _load_model()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {name: value.to(model.device) for name, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    input_length = inputs["input_ids"].shape[1]
    return tokenizer.decode(
        generated[0][input_length:],
        skip_special_tokens=True,
    ).strip()


def is_model_loaded() -> bool:
    """Report whether this process has successfully loaded the local model."""
    return _MODEL_LOADED
