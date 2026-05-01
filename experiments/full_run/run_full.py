"""Full evaluation on the LaMP val set (1500 examples) for the two tasks
where smoke runs showed positive persona effect: LaMP-2 and LaMP-7.

Three runs per call:
    1. Zero-shot baseline (control).
    2. Persona steering at the *optimal* layer (read from layer_search results;
       falls back to a hand-picked default if file not found).
    3. Persona steering at a *default* layer (the per-config default layer
       used in the original smoke runs) — for fair before/after comparison.

Usage:
    python run_full.py --model Qwen/Qwen3-8B  --task LaMP-2
    python run_full.py --model Qwen/Qwen3-14B --task LaMP-7
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
    get_decoder_layers,
)


DEFAULT_LAYERS = {
    "Qwen3-8B": 18,
    "Qwen3-14B": 20,
    "Mistral-Small-24B-Instruct-2501": 22,
    "Llama-3.1-8B-Instruct": 16,
}


def load_optimal_layer(model_name: str, task: str,
                       results_dir: Path) -> int | None:
    short = model_name.split("/")[-1]
    p = results_dir / "layer_search" / f"layer_search_{short}_{task}.json"
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
    return int(d["best_layer"]["layer_idx"])


def default_layer(model_name: str) -> int:
    return DEFAULT_LAYERS.get(model_name.split("/")[-1], 16)


@torch.no_grad()
def run_eval_loop(
    model, tokenizer, dataset, *,
    persona_vectors_per_user: list[torch.Tensor | None] | None,
    layer_idx: int | None,
    alpha: float,
    chat_kwargs: dict,
    system_prompt: str,
) -> tuple[list[str], list[str], float]:
    preds, refs = [], []
    t0 = time.time()
    for i, sample in enumerate(dataset):
        v = persona_vectors_per_user[i] if persona_vectors_per_user else None
        pred = persona_steered_generate(
            model, tokenizer,
            user_input=sample["input_text"],
            persona_vector=v, layer_idx=layer_idx, alpha=alpha,
            max_new_tokens=dataset.max_new_tokens,
            chat_kwargs=chat_kwargs, system_prompt=system_prompt,
        )
        preds.append(pred)
        refs.append(sample["output_text"].strip())
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = elapsed / (i + 1)
            eta = rate * (len(dataset) - (i + 1))
            print(f"  [{i+1}/{len(dataset)}] {elapsed:.0f}s rate={rate:.1f}s/ex eta={eta:.0f}s")
    return preds, refs, time.time() - t0


@torch.no_grad()
def extract_all_vectors(
    model, tokenizer, dataset, *,
    layer_idx: int, chat_kwargs: dict,
    extraction_questions: list[str],
) -> list[torch.Tensor | None]:
    pv = PersonaVectors(
        model=model, tokenizer=tokenizer, layer_idx=layer_idx,
        max_new_tokens=50, chat_template_kwargs=chat_kwargs,
    )
    out: list[torch.Tensor | None] = []
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
        out.append(v)
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = elapsed / (i + 1)
            eta = rate * (len(dataset) - (i + 1))
            print(f"  extract [{i+1}/{len(dataset)}] {elapsed:.0f}s rate={rate:.1f}s/ex eta={eta:.0f}s")
    return out


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--task", default="LaMP-2")
    ap.add_argument("--n_samples", type=int, default=1500)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="results/full_run")
    ap.add_argument("--skip_default_layer", action="store_true",
                    help="Skip the default-layer comparison run (saves ~33% time).")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    info = task_info(args.task)
    metric = info["metric"]

    optimal = load_optimal_layer(args.model, args.task, ROOT / "results")
    default = default_layer(args.model)
    chosen_layer = optimal if optimal is not None else default
    print(f"=== Full run: {args.model} on {args.task}, n={args.n_samples}, α={args.alpha} ===")
    print(f"  optimal layer (from layer_search): {optimal}")
    print(f"  default layer (config fallback):   {default}")
    print(f"  chosen for primary run:            {chosen_layer}")

    chat_kwargs = chat_kwargs_for(args.model)
    system_prompt = system_prompt_for(args.model)

    model, tokenizer = load_model_and_tokenizer(args.model)

    dataset = LaMPDataset(task=args.task, split="val", n_samples=args.n_samples,
                          data_dir=str(ROOT / "data"))
    extraction_questions = dataset.sample_train_inputs(k=1, seed=args.seed)

    # 1) Zero-shot
    print("\n--- Zero-shot ---")
    preds_zs, refs_zs, wall_zs = run_eval_loop(
        model, tokenizer, dataset,
        persona_vectors_per_user=None, layer_idx=None, alpha=0.0,
        chat_kwargs=chat_kwargs, system_prompt=system_prompt,
    )
    m_zs = compute_metric(metric, preds_zs, refs_zs)
    print(f"  zero-shot {metric}: {m_zs}")

    # 2) Persona @ optimal layer
    print(f"\n--- Persona α={args.alpha} @ layer {chosen_layer} (optimal) ---")
    print("  extracting per-user vectors...")
    vecs_opt = extract_all_vectors(
        model, tokenizer, dataset,
        layer_idx=chosen_layer, chat_kwargs=chat_kwargs,
        extraction_questions=extraction_questions,
    )
    preds_opt, refs_opt, wall_opt = run_eval_loop(
        model, tokenizer, dataset,
        persona_vectors_per_user=vecs_opt, layer_idx=chosen_layer, alpha=args.alpha,
        chat_kwargs=chat_kwargs, system_prompt=system_prompt,
    )
    m_opt = compute_metric(metric, preds_opt, refs_opt)
    print(f"  persona@{chosen_layer} {metric}: {m_opt}")

    # 3) Persona @ default layer (only if different from optimal)
    m_def = None
    wall_def = None
    if (not args.skip_default_layer) and (default != chosen_layer):
        print(f"\n--- Persona α={args.alpha} @ layer {default} (default) ---")
        vecs_def = extract_all_vectors(
            model, tokenizer, dataset,
            layer_idx=default, chat_kwargs=chat_kwargs,
            extraction_questions=extraction_questions,
        )
        preds_def, refs_def, wall_def = run_eval_loop(
            model, tokenizer, dataset,
            persona_vectors_per_user=vecs_def, layer_idx=default, alpha=args.alpha,
            chat_kwargs=chat_kwargs, system_prompt=system_prompt,
        )
        m_def = compute_metric(metric, preds_def, refs_def)
        print(f"  persona@{default} {metric}: {m_def}")

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"full_{args.model.split('/')[-1]}_{args.task}.json"
    payload = {
        "model": args.model, "task": args.task, "n_samples": args.n_samples,
        "alpha": args.alpha, "seed": args.seed,
        "metric": metric,
        "optimal_layer": optimal, "default_layer": default, "chosen_layer": chosen_layer,
        "zero_shot": {"value": m_zs, "wall_seconds": wall_zs, "num_eval": len(refs_zs),
                      "sample_preds": preds_zs[:5], "sample_refs": refs_zs[:5]},
        "persona_optimal": {"layer": chosen_layer, "value": m_opt,
                            "wall_seconds": wall_opt, "num_eval": len(refs_opt),
                            "sample_preds": preds_opt[:5], "sample_refs": refs_opt[:5]},
        "persona_default": (None if m_def is None else {
            "layer": default, "value": m_def, "wall_seconds": wall_def,
        }),
        "timestamp": datetime.now().isoformat(),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
