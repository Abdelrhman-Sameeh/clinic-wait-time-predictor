"""Cached local Qwen generation for the clinic RAG pipeline."""

from typing import Any

import streamlit as st


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 192
_MODEL_LOADED = False


@st.cache_resource(show_spinner="Loading the local clinic assistant…")
def _load_model() -> tuple[Any, Any]:
    """Load the tokenizer and model once per process."""
    global _MODEL_LOADED
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto",
    )
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
