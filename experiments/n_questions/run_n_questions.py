"""Ablation: persona-vector quality vs number of extraction questions.

n_questions ∈ {1, 3, 5, 10, 20}. For each setting, build per-user artifacts
with k randomly-sampled train inputs as extraction questions, extract a
vector at the optimal layer, run inference, record accuracy.

Hypothesis: more questions → less noisy persona vector → higher accuracy
(plateau around k≈10, matching the 20-question count from the original
Anthropic paper §3).

Usage:
    python run_n_questions.py --model Qwen/Qwen3-8B --task LaMP-2 --layer_idx 16
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


def load_optimal_layer(model_name: str, task: str,
                       results_dir: Path, fallback: int) -> int:
    short = model_name.split("/")[-1]
    p = results_dir / "layer_search" / f"layer_search_{short}_{task}.json"
    if p.exists():
        with open(p) as f:
            return int(json.load(f)["best_layer"]["layer_idx"])
    return fallback


@torch.no_grad()
def run_one_setting(
    model, tokenizer, dataset, *,
    layer_idx: int, alpha: float,
    n_questions: int, chat_kwargs: dict, system_prompt: str, seed: int,
) -> dict:
    pv = PersonaVectors(
        model=model, tokenizer=tokenizer, layer_idx=layer_idx,
        max_new_tokens=50, chat_template_kwargs=chat_kwargs,
    )
    extraction_questions = dataset.sample_train_inputs(k=n_questions, seed=seed)

    preds, refs = [], []
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
    return {"preds": preds, "refs": refs, "wall_seconds": time.time() - t0}


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--task", default="LaMP-2")
    ap.add_argument("--n_samples", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--layer_idx", type=int, default=None,
                    help="Override; otherwise reads layer_search results.")
    ap.add_argument("--n_questions_grid", type=int, nargs="+",
                    default=[1, 3, 5, 10, 20])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="results/n_questions")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    info = task_info(args.task)
    metric = info["metric"]
    chat_kwargs = chat_kwargs_for(args.model)
    system_prompt = system_prompt_for(args.model)

    layer_idx = args.layer_idx if args.layer_idx is not None else \
        load_optimal_layer(args.model, args.task, ROOT / "results", fallback=16)
    print(f"=== n_questions ablation: {args.model} / {args.task} / layer {layer_idx} ===")

    model, tokenizer = load_model_and_tokenizer(args.model)
    dataset = LaMPDataset(task=args.task, split="val", n_samples=args.n_samples,
                          data_dir=str(ROOT / "data"))

    results = []
    for k in args.n_questions_grid:
        print(f"\n--- n_questions={k} ---")
        run = run_one_setting(
            model, tokenizer, dataset,
            layer_idx=layer_idx, alpha=args.alpha,
            n_questions=k, chat_kwargs=chat_kwargs,
            system_prompt=system_prompt, seed=args.seed,
        )
        m = compute_metric(metric, run["preds"], run["refs"])
        print(f"  k={k}: {m}")
        results.append({
            "n_questions": k, "metric": metric, "value": m,
            "wall_seconds": run["wall_seconds"], "num_eval": len(run["refs"]),
        })

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"n_questions_{args.model.split('/')[-1]}_{args.task}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model, "task": args.task, "n_samples": args.n_samples,
            "alpha": args.alpha, "layer_idx": layer_idx, "seed": args.seed,
            "results": results, "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
