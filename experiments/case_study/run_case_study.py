"""Single-user case study: trace prediction trajectory across α values.

The mech-interp story for the paper needs one concrete, memorable example.
We pick one LaMP-2 user with a distinctive profile and show how the model's
top-k next-token probabilities at the answer position shift as α grows from
0 (no steering) through the optimal α=1 to over-steered α=2.

Output:
    - results/case_study/case_<task>_<user>.json (per-α top-k probs, predictions,
      facts text, profile, gold)
    - figures/fig_case_study_<task>_<user>.pdf (multi-panel)

Usage:
    python experiments/case_study/run_case_study.py \\
        --model Qwen/Qwen3-8B --task LaMP-2 --layer_idx 13 \\
        --user_idx 1 --alphas 0.0 0.5 1.0 1.5 2.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import (
    FactExtractor, LaMPDataset, PersonaSteering, PersonaVectors,
    chat_kwargs_for, load_model_and_tokenizer, system_prompt_for,
    task_info,
)


@torch.no_grad()
def first_token_topk(
    model, tokenizer, *, prompt: str, k: int = 10,
    persona_vector: torch.Tensor | None = None,
    layer_idx: int | None = None, alpha: float = 0.0,
) -> tuple[list[tuple[str, float]], str]:
    """Return (top-k tokens with probabilities, decoded greedy continuation up to first newline)."""
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024
                    ).to(next(model.parameters()).device)

    if persona_vector is not None and abs(alpha) > 0.0:
        ctx = PersonaSteering(model, layer_idx).hook(persona_vector, alpha=alpha)
    else:
        from contextlib import nullcontext
        ctx = nullcontext()

    with ctx:
        outputs = model(**enc)
        logits = outputs.logits[0, -1, :]
        probs = F.softmax(logits.float(), dim=-1)
        topk_p, topk_idx = torch.topk(probs, k=k)
        topk = [(tokenizer.decode([int(i)]).strip(), float(p))
                for p, i in zip(topk_p, topk_idx)]

        # Greedy continuation (so we have a "prediction" too)
        greedy_out = model.generate(
            **enc, max_new_tokens=8, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = greedy_out[0, enc["input_ids"].shape[1]:]
        pred = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return topk, pred


def build_chat_prompt(tokenizer, user_input: str, system_prompt: str | None,
                      chat_kwargs: dict) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_input})
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **chat_kwargs,
        )
    return user_input


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--task", default="LaMP-2")
    ap.add_argument("--layer_idx", type=int, default=13)
    ap.add_argument("--user_idx", type=int, default=1,
                    help="Index into the dedup'd unique-user list.")
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--variant", choices=["template", "fact"], default="fact")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="results/case_study")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    chat_kwargs = chat_kwargs_for(args.model)
    system_prompt = system_prompt_for(args.model)

    print(f"=== Case study: {args.model} on {args.task} user_idx={args.user_idx} ===")
    print(f"  layer={args.layer_idx} variant={args.variant} αs={args.alphas}")

    model, tokenizer = load_model_and_tokenizer(args.model)
    dataset = LaMPDataset(task=args.task, split="val", n_samples=args.user_idx + 1,
                          data_dir=str(ROOT / "data"), unique_users=True)
    samples = list(dataset)
    if args.user_idx >= len(samples):
        raise IndexError(f"Only {len(samples)} unique users available")
    s = samples[args.user_idx]

    # Get the user's persona vector
    if args.variant == "fact":
        extractor = FactExtractor(model=model, tokenizer=tokenizer, task=args.task)
        facts = extractor.extract_facts(s["behavior_profile_text"])
        positive = [facts["positive_prompt"]]
        negative = [facts["negative_prompt"]]
        facts_text = facts["raw_facts"]
    else:
        positive = s["positive_system_prompts"]
        negative = s["negative_system_prompts"]
        facts_text = "(template path — no extracted facts)"

    pv = PersonaVectors(
        model=model, tokenizer=tokenizer, layer_idx=args.layer_idx,
        max_new_tokens=50, chat_template_kwargs=chat_kwargs,
    )
    print("Extracting persona vector for chosen user...")
    vector = pv.extract(
        positive_prompts=positive,
        negative_prompts=negative,
        extraction_questions=[s["input_text"]],
    ).to(model.device)

    prompt = build_chat_prompt(tokenizer, s["input_text"], system_prompt, chat_kwargs)
    gold = s["output_text"].strip()

    rows = []
    for alpha in args.alphas:
        topk, pred = first_token_topk(
            model, tokenizer, prompt=prompt, k=10,
            persona_vector=vector if alpha != 0 else None,
            layer_idx=args.layer_idx, alpha=alpha,
        )
        match = (pred.lower() == gold.lower())
        print(f"  α={alpha:>4} pred={pred!r:>20}  match={match}  top-5: "
              + ", ".join(f"{t!r}={p:.3f}" for t, p in topk[:5]))
        rows.append({
            "alpha": alpha, "prediction": pred, "match_gold": match,
            "topk": topk,
        })

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    user_tag = f"user{args.user_idx:03d}_{args.variant}"
    out_path = out_dir / f"case_{args.task}_{user_tag}.json"

    payload = {
        "model": args.model, "task": args.task,
        "layer_idx": args.layer_idx, "user_idx": args.user_idx,
        "variant": args.variant,
        "alphas": args.alphas,
        "gold": gold,
        "input_text": s["input_text"],
        "facts_text": facts_text,
        "profile": s["behavior_profile_text"],
        "vector_norm": float(vector.float().norm().cpu()),
        "vector_first10": vector[:10].float().cpu().tolist(),
        "rows": rows,
        "timestamp": datetime.now().isoformat(),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
