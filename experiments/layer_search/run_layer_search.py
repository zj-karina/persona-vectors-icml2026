"""Layer Search: find the best persona-vector extraction layer per LLM.

For each model: sweep the middle 60% of layers (skip first and last 20%) at
stride 2, run persona steering on LaMP-2 with n=200 examples, save accuracy
vs layer index. We pick LaMP-2 because that's where smoke-runs showed the
biggest persona effect (+10 pp), so the signal-to-noise ratio is best for
finding the optimal layer.

Output: results/layer_search/layer_search_<model>_<task>.json with all
per-layer accuracies plus the argmax in `best_layer`.

Usage:
    python run_layer_search.py --model Qwen/Qwen3-8B --task LaMP-2
    python run_layer_search.py --model Qwen/Qwen3-14B --task LaMP-2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import (
    LaMPDataset, PersonaVectors, compute_metric, load_model_and_tokenizer,
    persona_steered_generate, chat_kwargs_for, system_prompt_for, task_info,
    get_decoder_layers,
)


def get_layer_grid(model, stride: int = 2) -> list[int]:
    """Middle 60% of layers (skip first and last 20%), at given stride."""
    n = len(get_decoder_layers(model))
    start = int(n * 0.2)
    end = int(n * 0.8)
    return list(range(start, end, stride))


def evaluate_with_layer(
    model, tokenizer, dataset, *,
    layer_idx: int,
    alpha: float,
    chat_kwargs: dict,
    system_prompt: str,
    extraction_questions: list[str],
) -> tuple[list[str], list[str], float]:
    """Extract per-user vector at layer_idx, generate predictions, return preds, refs, wall."""
    pv = PersonaVectors(
        model=model, tokenizer=tokenizer, layer_idx=layer_idx,
        max_new_tokens=50, chat_template_kwargs=chat_kwargs,
    )

    preds: list[str] = []
    refs: list[str] = []
    t0 = time.time()
    for i, sample in enumerate(dataset):
        try:
            vector = pv.extract(
                positive_prompts=sample["positive_system_prompts"],
                negative_prompts=sample["negative_system_prompts"],
                extraction_questions=extraction_questions,
            )
        except RuntimeError:
            vector = None

        pred = persona_steered_generate(
            model, tokenizer,
            user_input=sample["input_text"],
            persona_vector=vector,
            layer_idx=layer_idx,
            alpha=alpha,
            max_new_tokens=dataset.max_new_tokens,
            chat_kwargs=chat_kwargs,
            system_prompt=system_prompt,
        )
        preds.append(pred)
        refs.append(sample["output_text"].strip())
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"  layer {layer_idx}: [{i+1}/{len(dataset)}] {elapsed:.0f}s")
    return preds, refs, time.time() - t0


@torch.no_grad()
def run_layer_search(
    model_name: str,
    task: str = "LaMP-2",
    n_samples: int = 200,
    alpha: float = 1.0,
    stride: int = 2,
    output_dir: str = "results/layer_search",
    seed: int = 42,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print(f"=== Layer search: {model_name} on {task}, n={n_samples}, α={alpha} ===")
    chat_kwargs = chat_kwargs_for(model_name)
    system_prompt = system_prompt_for(model_name)

    model, tokenizer = load_model_and_tokenizer(model_name)
    layers = get_layer_grid(model, stride=stride)
    print(f"Layers to sweep ({len(layers)}): {layers}")

    info = task_info(task)
    dataset = LaMPDataset(task=task, split="val", n_samples=n_samples,
                          data_dir=str(ROOT / "data"))
    extraction_questions = dataset.sample_train_inputs(k=1, seed=seed)
    metric = info["metric"]

    # Zero-shot baseline (control) — no steering, single forward.
    print(f"\n--- Zero-shot baseline ---")
    preds_zs, refs_zs = [], []
    t0 = time.time()
    for i, s in enumerate(dataset):
        pred = persona_steered_generate(
            model, tokenizer,
            user_input=s["input_text"],
            persona_vector=None, layer_idx=None, alpha=0.0,
            max_new_tokens=dataset.max_new_tokens,
            chat_kwargs=chat_kwargs, system_prompt=system_prompt,
        )
        preds_zs.append(pred)
        refs_zs.append(s["output_text"].strip())
        if (i + 1) % 25 == 0:
            print(f"  zs [{i+1}/{len(dataset)}] {time.time()-t0:.0f}s")
    zs_metric = compute_metric(metric, preds_zs, refs_zs)
    print(f"  zero-shot {metric}: {zs_metric}")

    results = []
    for layer_idx in layers:
        print(f"\n--- Layer {layer_idx} ---")
        preds, refs, wall = evaluate_with_layer(
            model, tokenizer, dataset,
            layer_idx=layer_idx, alpha=alpha,
            chat_kwargs=chat_kwargs, system_prompt=system_prompt,
            extraction_questions=extraction_questions,
        )
        m = compute_metric(metric, preds, refs)
        n_layers_total = len(get_decoder_layers(model))
        results.append({
            "layer_idx": layer_idx,
            "layer_fraction": layer_idx / n_layers_total,
            "metric": metric,
            "value": m,
            "wall_seconds": wall,
        })
        print(f"  layer {layer_idx}: {m}")

    # Pick best by primary metric (accuracy maxed; mae minimized; rouge maxed).
    def primary(v):
        if metric == "accuracy":
            return v["value"]["accuracy"]
        if metric == "regression":
            return -v["value"]["mae"]
        if metric == "rouge":
            return v["value"]["ROUGE-L"]
        return 0.0

    best = max(results, key=primary)
    print(f"\n=== Best layer: {best['layer_idx']} ({best['layer_fraction']:.1%}) ===")
    print(f"    metric: {best['value']}")

    out_path = Path(output_dir) / f"layer_search_{model_name.split('/')[-1]}_{task}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": model_name,
            "task": task,
            "n_samples": n_samples,
            "alpha": alpha,
            "stride": stride,
            "seed": seed,
            "layers_tested": layers,
            "n_layers_total": len(get_decoder_layers(model)),
            "zero_shot": {"metric": metric, "value": zs_metric, "num_eval": len(refs_zs)},
            "results": results,
            "best_layer": best,
        }, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--task", default="LaMP-2")
    ap.add_argument("--n_samples", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--output_dir", default="results/layer_search")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run_layer_search(
        args.model, args.task, args.n_samples, args.alpha,
        stride=args.stride, output_dir=args.output_dir, seed=args.seed,
    )
