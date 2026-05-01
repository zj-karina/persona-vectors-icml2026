"""Scan multiple users to pick the most demonstrative case-study target.

We're looking for users where:
  - α=0 prediction differs from α=1 (or 2) prediction, OR
  - gold-token probability shifts significantly across α.

Reuses the fact_cache from the positive-control run on LaMP-2 to skip
fact extraction. Per-user cost: ~1 sec × 5 alphas = ~5 sec.

Output: results/case_study/scan_<task>.json with one row per user_idx.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import (
    FactExtractor, LaMPDataset, PersonaSteering, PersonaVectors,
    chat_kwargs_for, load_model_and_tokenizer, system_prompt_for,
)


@torch.no_grad()
def topk_at_first(model, tokenizer, prompt, k=5, vector=None, layer_idx=None, alpha=0.0):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024
                   ).to(next(model.parameters()).device)
    if vector is not None and abs(alpha) > 0:
        ctx = PersonaSteering(model, layer_idx).hook(vector, alpha=alpha)
    else:
        from contextlib import nullcontext
        ctx = nullcontext()
    with ctx:
        out = model(**enc)
        probs = torch.nn.functional.softmax(out.logits[0, -1].float(), dim=-1)
        topk_p, topk_i = torch.topk(probs, k)
        gen = model.generate(**enc, max_new_tokens=4, do_sample=False,
                            pad_token_id=tokenizer.pad_token_id)
        new_tokens = gen[0, enc["input_ids"].shape[1]:]
        pred = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return [(tokenizer.decode([int(i)]).strip(), float(p))
            for p, i in zip(topk_p, topk_i)], pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--task", default="LaMP-2")
    ap.add_argument("--layer_idx", type=int, default=13)
    ap.add_argument("--n_users", type=int, default=30)
    ap.add_argument("--variant", choices=["template", "fact"], default="fact")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    args = ap.parse_args()

    chat_kwargs = chat_kwargs_for(args.model)
    system_prompt = system_prompt_for(args.model)

    model, tokenizer = load_model_and_tokenizer(args.model)
    dataset = LaMPDataset(task=args.task, split="val", n_samples=args.n_users,
                          data_dir=str(ROOT / "data"), unique_users=True)
    samples = list(dataset)

    if args.variant == "fact":
        # reuse fact cache from positive_control
        cache_path = ROOT / "results" / "positive_control" / f"cache_facts_{args.task}.json"
        if cache_path.exists():
            with open(cache_path) as f:
                cache = json.load(f)
            fact_extractor = FactExtractor(model, tokenizer, args.task)
        else:
            print(f"WARNING: {cache_path} missing — falling back to template variant")
            args.variant = "template"
            cache = {}
    else:
        cache = {}

    pv = PersonaVectors(model=model, tokenizer=tokenizer, layer_idx=args.layer_idx,
                        max_new_tokens=50, chat_template_kwargs=chat_kwargs)

    rows = []
    for i, s in enumerate(samples):
        # Build positive/negative prompts
        if args.variant == "fact":
            user_id = str(s.get("id") or i)
            if user_id not in cache:
                facts = FactExtractor(model, tokenizer, args.task).extract_facts(
                    s["behavior_profile_text"])
                cache[user_id] = facts
            facts = cache[user_id]
            positive = [facts["positive_prompt"]]
            negative = [facts["negative_prompt"]]
        else:
            positive = s["positive_system_prompts"]
            negative = s["negative_system_prompts"]

        # Build chat prompt
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": s["input_text"]})
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True,
                                               **chat_kwargs)

        # Extract vector
        try:
            v = pv.extract(positive, negative, [s["input_text"]]).to(model.device)
        except RuntimeError:
            print(f"  user {i}: extract failed — skipping")
            continue

        gold = s["output_text"].strip().lower()
        per_alpha = {}
        preds_seen = set()
        for alpha in args.alphas:
            topk, pred = topk_at_first(model, tokenizer, prompt, k=5,
                                        vector=v if alpha != 0 else None,
                                        layer_idx=args.layer_idx, alpha=alpha)
            per_alpha[alpha] = {"pred": pred, "topk": topk,
                                "match": pred.lower() == gold}
            preds_seen.add(pred.lower())

        # "interestingness": more distinct predictions across α = better case study
        interestingness = len(preds_seen)
        rows.append({
            "user_idx": i, "gold": gold,
            "input": s["input_text"][:120],
            "n_distinct_preds": interestingness,
            "preds": {a: p["pred"] for a, p in per_alpha.items()},
            "matches": {a: p["match"] for a, p in per_alpha.items()},
            "P(gold) by α": {
                a: next((p for t, p in v["topk"] if t.lower() == gold), 0.0)
                for a, v in per_alpha.items()
            },
        })
        print(f"  user {i:>3} gold={gold!r:>20} preds={list(per_alpha.values())[0]['pred']!r}->"
              f"{list(per_alpha.values())[1]['pred']!r}->{list(per_alpha.values())[-1]['pred']!r}  "
              f"distinct={interestingness}")

    # Pick top users by interestingness
    rows.sort(key=lambda r: (-r["n_distinct_preds"], r["user_idx"]))
    print("\n=== Top candidates for case study ===")
    for r in rows[:5]:
        print(f"  user_idx={r['user_idx']} gold={r['gold']!r} "
              f"distinct={r['n_distinct_preds']}  preds={r['preds']}  "
              f"P(gold)={r['P(gold) by α']}")

    out_dir = ROOT / "results" / "case_study"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"scan_{args.task}_{args.variant}.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
