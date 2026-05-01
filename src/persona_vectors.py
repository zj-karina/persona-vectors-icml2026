"""Persona Vectors: paper-faithful extraction + steering for per-user identities.

Implements the algorithm from Rimsky et al. (Anthropic, arXiv:2507.21509),
adapted for *per-user* personalization rather than global character traits.

Two classes:
    PersonaVectors  — extract a vector at a given residual-stream layer from
                      a (positive_prompts, negative_prompts, questions) bundle.
    PersonaSteering — register a forward hook that adds α·v to that layer.

API designed for the ICML 2026 mech-interp paper experiments. Keep it simple:
no judges, no filtering — pass artifacts in, get a vector out.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Sequence

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Layer access (works for Llama / Qwen / Mistral / Gemma2 architectures)
# ---------------------------------------------------------------------------


def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    base = getattr(model, "model", model)
    if hasattr(base, "layers"):
        return base.layers
    if hasattr(base, "model") and hasattr(base.model, "layers"):
        return base.model.layers
    raise AttributeError(f"Cannot locate decoder layers on {type(model).__name__}")


def _layer_hidden(out):
    return out[0] if isinstance(out, tuple) else out


def _replace_layer_hidden(out, new_hidden):
    if isinstance(out, tuple):
        return (new_hidden, *out[1:])
    return new_hidden


# ---------------------------------------------------------------------------
# PersonaVectors — extraction
# ---------------------------------------------------------------------------


class PersonaVectors:
    """Extract a per-user persona vector at a single residual-stream layer.

    Algorithm (paper-faithful, Anthropic 2025):
        1. For each (positive system_prompt, question) pair:
             generate response, take mean residual-stream activation at
             `layer_idx` over the response tokens.
        2. Same for negative system_prompts.
        3. v = mean(positive_activations) - mean(negative_activations).

    Args:
        model: HF AutoModelForCausalLM (Qwen3 / Llama / Mistral / ...).
        tokenizer: matched tokenizer.
        layer_idx: which decoder block to extract from (0-indexed).
        system_prompt: optional default; per-call positive/negative override.
        max_new_tokens: response length used for activation pooling.
        device: defaults to next(model.parameters()).device.
        chat_template_kwargs: e.g. {"enable_thinking": False} for Qwen3.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        layer_idx: int,
        system_prompt: str | None = None,
        max_new_tokens: int = 50,
        device: torch.device | str | None = None,
        chat_template_kwargs: dict | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.chat_template_kwargs = chat_template_kwargs or {}

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        n_layers = len(get_decoder_layers(self.model))
        if not (0 <= layer_idx < n_layers):
            raise ValueError(f"layer_idx {layer_idx} out of range [0, {n_layers})")

    # ----- helpers --------------------------------------------------------

    def _format_chat(self, system: str, user: str) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True,
                **self.chat_template_kwargs,
            )
        return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"

    @torch.no_grad()
    def _generate_and_pool(self, system: str, question: str) -> torch.Tensor | None:
        """Generate response, return mean activation at layer_idx over response
        tokens. Returns None if response is empty."""
        prompt = self._format_chat(system, question)
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = enc["input_ids"].shape[1]

        out = self.model.generate(
            **enc,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        full_ids = out[0].unsqueeze(0)
        if full_ids.shape[1] <= prompt_len:
            return None

        # Single forward to grab hidden states (cheaper than caching during generate).
        outputs = self.model(full_ids, output_hidden_states=True, use_cache=False)
        hs = outputs.hidden_states[self.layer_idx + 1][0]  # [seq_len, hidden]
        response_hs = hs[prompt_len:]
        if response_hs.numel() == 0:
            return None
        return response_hs.mean(dim=0).detach().float().cpu()

    # ----- public API -----------------------------------------------------

    def extract(
        self,
        positive_prompts: Sequence[str],
        negative_prompts: Sequence[str],
        extraction_questions: Sequence[str],
    ) -> torch.Tensor:
        """Return a single persona vector at self.layer_idx.

        Shape: [hidden_dim]. Uses `mean(positive) - mean(negative)`.
        """
        pos_acts: list[torch.Tensor] = []
        neg_acts: list[torch.Tensor] = []

        for sp in positive_prompts:
            for q in extraction_questions:
                v = self._generate_and_pool(sp, q)
                if v is not None:
                    pos_acts.append(v)

        for sp in negative_prompts:
            for q in extraction_questions:
                v = self._generate_and_pool(sp, q)
                if v is not None:
                    neg_acts.append(v)

        if not pos_acts or not neg_acts:
            raise RuntimeError(
                f"Empty bucket: pos={len(pos_acts)} neg={len(neg_acts)} — "
                "model may have produced no response tokens"
            )
        return torch.stack(pos_acts).mean(dim=0) - torch.stack(neg_acts).mean(dim=0)


# ---------------------------------------------------------------------------
# PersonaSteering — inference-time injection
# ---------------------------------------------------------------------------


class PersonaSteering:
    """Forward-hook injector: add `alpha * vector` to layer_idx residual stream.

    Use as a context manager:
        steering = PersonaSteering(model, layer_idx=16)
        with steering.hook(vector, alpha=1.0):
            model.generate(...)
    """

    def __init__(self, model: nn.Module, layer_idx: int):
        self.model = model
        self.layer_idx = layer_idx
        self._layers = get_decoder_layers(model)
        n_layers = len(self._layers)
        if not (0 <= layer_idx < n_layers):
            raise ValueError(f"layer_idx {layer_idx} out of range [0, {n_layers})")

    @contextmanager
    def hook(self, vector: torch.Tensor, alpha: float = 1.0, position: str = "all"):
        """Register a forward hook on layer self.layer_idx that adds α·v to
        every residual-stream position (or last token only if position='last').
        """
        if position not in ("all", "last"):
            raise ValueError("position must be 'all' or 'last'")

        v_cpu = vector.detach().float().cpu()
        cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

        def vec_for(device, dtype):
            key = (device, dtype)
            if key not in cache:
                cache[key] = v_cpu.to(device=device, dtype=dtype)
            return cache[key]

        def fwd_hook(module, inputs, output):
            hidden = _layer_hidden(output)
            v = vec_for(hidden.device, hidden.dtype)
            if position == "all":
                hidden = hidden + alpha * v
            else:
                hidden = hidden.clone()
                hidden[..., -1, :] = hidden[..., -1, :] + alpha * v
            return _replace_layer_hidden(output, hidden)

        handle = self._layers[self.layer_idx].register_forward_hook(fwd_hook)
        try:
            yield self
        finally:
            handle.remove()


# ---------------------------------------------------------------------------
# PersonaMonitor — projection score (no generation needed)
# ---------------------------------------------------------------------------


class PersonaMonitor:
    """Compute hidden_state @ persona_vector at the last prompt token, layer L."""

    def __init__(self, model: nn.Module, layer_idx: int):
        self.model = model
        self.layer_idx = layer_idx

    @torch.no_grad()
    def score(self, vector: torch.Tensor, input_ids: torch.Tensor,
              attention_mask: torch.Tensor | None = None,
              normalize: bool = True) -> torch.Tensor:
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         output_hidden_states=True, use_cache=False)
        hs = out.hidden_states[self.layer_idx + 1]

        if attention_mask is None:
            last_idx = torch.full((input_ids.size(0),), input_ids.size(1) - 1,
                                  dtype=torch.long, device=device)
        else:
            last_idx = attention_mask.long().sum(dim=1) - 1
        batch_idx = torch.arange(input_ids.size(0), device=device)
        last_h = hs[batch_idx, last_idx].float()

        v = vector.to(device=device, dtype=last_h.dtype)
        if normalize:
            v = v / (v.norm() + 1e-8)
            last_h = last_h / (last_h.norm(dim=-1, keepdim=True) + 1e-8)
        return (last_h * v).sum(dim=-1).cpu()
