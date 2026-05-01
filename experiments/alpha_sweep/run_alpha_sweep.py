"""α (steering strength) sweep on LaMP-2 with n=200, optimal layer.

Why: in the smoke run α∈{0.5, 1.0, 1.5, 2.0} all collapsed to nearly identical
predictions, suggesting α may simply be too small relative to residual stream
norm. We test α∈{0, 0.5, 1, 2, 4, 8, 16} to map the steering response curve.

Large α may degrade base capability (paper §6) — we expect a peak then decay.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import (
    LaMPDataset, PersonaVectors, compute_metric, load_model_and_tokenizer,
    persona_steered_generate, chat_kwargs_for, system_prompt_for, task_info,
)


def load_optimal_layer(model_name, task, results_dir, fallback=16):
    short = model_name.split("/")[-1]
    p = results_dir / "layer_search" / f"layer_search_{short}_{task}.json"
    if p.exists():
        with open(p) as f:
            return int(json.load(f)["best_layer"]["layer_idx"])
    return fallback


@torch.no_grad()
def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--task", default="LaMP-2")
    ap.add_argument("--n_samples", type=int, default=200)
    ap.add_argument("--layer_idx", type=int, default=None)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="results/alpha_sweep")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    info = task_info(args.task)
    metric = info["metric"]
    chat_kwargs = chat_kwargs_for(args.model)
    system_prompt = system_prompt_for(args.model)

    layer_idx = args.layer_idx if args.layer_idx is not None else \
        load_optimal_layer(args.model, args.task, ROOT / "results")

    print(f"=== α-sweep: {args.model} / {args.task} / layer {layer_idx} ===")
    model, tokenizer = load_model_and_tokenizer(args.model)
    dataset = LaMPDataset(task=args.task, split="val", n_samples=args.n_samples,
                          data_dir=str(ROOT / "data"))
    extraction_questions = dataset.sample_train_inputs(k=1, seed=args.seed)

    # Extract per-user vectors ONCE — reuse across all alphas (vectors don't depend on α).
    pv = PersonaVectors(model=model, tokenizer=tokenizer, layer_idx=layer_idx,
                        max_new_tokens=50, chat_template_kwargs=chat_kwargs)
    print("Extracting per-user vectors (shared across α values)...")
    user_vectors: list[torch.Tensor | None] = []
    t0 = time.time()
    for i, s in enumerate(dataset):
        try:
            v = pv.extract(
                positive_prompts=s["positive_system_prompts"],
                negative_prompts=s["negative_system_prompts"],
                extraction_questions=extraction_questions,
            )
        except RuntimeError:
            v = None
        user_vectors.append(v)
        if (i + 1) % 50 == 0:
            print(f"  extract [{i+1}/{len(dataset)}] {time.time()-t0:.0f}s")

    results = []
    for alpha in args.alphas:
        print(f"\n--- α={alpha} ---")
        preds, refs = [], []
        t0 = time.time()
        for i, s in enumerate(dataset):
            v = user_vectors[i]
            pred = persona_steered_generate(
                model, tokenizer,
                user_input=s["input_text"],
                persona_vector=v, layer_idx=layer_idx, alpha=alpha,
                max_new_tokens=dataset.max_new_tokens,
                chat_kwargs=chat_kwargs, system_prompt=system_prompt,
            )
            preds.append(pred)
            refs.append(s["output_text"].strip())
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(dataset)}] {time.time()-t0:.0f}s")
        m = compute_metric(metric, preds, refs)
        print(f"  α={alpha}: {m}")
        results.append({"alpha": alpha, "metric": metric, "value": m,
                        "wall_seconds": time.time() - t0,
                        "sample_preds": preds[:5], "sample_refs": refs[:5]})

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"alpha_sweep_{args.model.split('/')[-1]}_{args.task}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model, "task": args.task, "n_samples": args.n_samples,
            "layer_idx": layer_idx, "seed": args.seed,
            "alphas": args.alphas, "results": results,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
