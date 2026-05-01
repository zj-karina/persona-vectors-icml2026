"""Shared inference helpers — model load, persona-steered generate, eval loop."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .persona_vectors import PersonaSteering, PersonaVectors


def is_qwen3(model_name: str) -> bool:
    return "qwen3" in model_name.lower() or "qwen3.5" in model_name.lower()


def load_model_and_tokenizer(
    model_name: str,
    torch_dtype: torch.dtype = torch.float16,
    device_map: str | dict | None = "auto",
    attn_implementation: str = "sdpa",
):
    tok_kwargs = {}
    if "mistral" in model_name.lower():
        tok_kwargs["fix_mistral_regex"] = True
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, tokenizer


def chat_kwargs_for(model_name: str) -> dict:
    """Return chat_template kwargs that disable Qwen3 thinking mode."""
    if is_qwen3(model_name):
        return {"enable_thinking": False}
    return {}


def system_prompt_for(model_name: str) -> str:
    """Default system prompt; for Qwen3 includes /no_think to skip CoT."""
    if is_qwen3(model_name):
        return "You are a helpful assistant. /no_think"
    return "You are a helpful assistant."


@torch.no_grad()
def persona_steered_generate(
    model,
    tokenizer,
    user_input: str,
    *,
    persona_vector: torch.Tensor | None = None,
    layer_idx: int | None = None,
    alpha: float = 1.0,
    max_new_tokens: int = 32,
    max_input_len: int = 1024,
    chat_kwargs: dict | None = None,
    system_prompt: str | None = None,
) -> str:
    """Generate one LaMP answer; if persona_vector & alpha != 0, apply steering."""
    chat_kwargs = chat_kwargs or {}

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_input})

    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **chat_kwargs,
        )
    else:
        prompt = user_input

    enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                    max_length=max_input_len).to(next(model.parameters()).device)

    if persona_vector is not None and abs(alpha) > 0.0 and layer_idx is not None:
        steering = PersonaSteering(model, layer_idx=layer_idx)
        ctx = steering.hook(persona_vector, alpha=alpha, position="all")
    else:
        ctx = nullcontext()

    with ctx:
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out[0, enc["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
