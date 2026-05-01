"""Positive control: template-based vs fact-based persona vectors.

Tests whether the rank-2 collapse of template-based vectors (Section "Phase 2"
of the main results) is due to the *artifact construction* — i.e., the fixed
positive-prompt template — rather than a fundamental property of the residual
stream geometry.

Construction:
    A) Template path  — uses the boilerplate positive prompts the dataset
                        already builds from the user's profile.
    B) Fact-based     — same model produces 5 specific facts per user from
                        their LaMP profile, and those facts become the
                        positive system prompt.

We compare:
    - Geometry: pairwise cosine, PCA(2), 90%-variance rank.
    - Downstream accuracy: zero-shot vs steered (both methods).

Same Qwen3-8B is used for fact extraction, vector extraction, and downstream
inference. No external APIs.

Usage:
    python experiments/positive_control/run_positive_control.py \
        --model Qwen/Qwen3-8B --task LaMP-2 --layer_idx 13 --n_users 30
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
    FactExtractor, LaMPDataset, PersonaSteering, PersonaVectors,
    chat_kwargs_for, compute_metric, get_decoder_layers,
    load_model_and_tokenizer, persona_steered_generate, system_prompt_for,
    task_info,
)


# ---------------------------------------------------------------------------
# Vector extraction across N users
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_vectors(
    model, tokenizer, samples, *,
    layer_idx: int,
    positive_key: str,
    negative_key: str,
    chat_kwargs: dict,
) -> np.ndarray:
    pv = PersonaVectors(
        model=model, tokenizer=tokenizer, layer_idx=layer_idx,
        max_new_tokens=50, chat_template_kwargs=chat_kwargs,
    )
    out = []
    t0 = time.time()
    for i, s in enumerate(samples):
        try:
            v = pv.extract(
                positive_prompts=s[positive_key],
                negative_prompts=s[negative_key],
                extraction_questions=[s["input_text"]],
            )
            out.append(v.cpu().float().numpy())
        except RuntimeError as e:
            print(f"  user {i}: extract failed ({e}) — skipping")
            out.append(np.zeros(model.config.hidden_size, dtype=np.float32))
        if (i + 1) % 5 == 0:
            print(f"  vectors {i+1}/{len(samples)} ({time.time()-t0:.0f}s)")
    return np.stack(out, axis=0)


# ---------------------------------------------------------------------------
# Geometry analysis
# ---------------------------------------------------------------------------


def compute_geometry(vectors: np.ndarray, label: str) -> dict:
    from sklearn.decomposition import PCA

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.maximum(norms, 1e-8)
    cos = normalized @ normalized.T
    off_diag = cos[~np.eye(len(cos), dtype=bool)]

    pca = PCA().fit(vectors)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    rank_90 = int(np.searchsorted(cumvar, 0.90)) + 1
    rank_95 = int(np.searchsorted(cumvar, 0.95)) + 1
    var_2pc = float(sum(pca.explained_variance_ratio_[:2]))

    verdict = (
        "TEMPLATE COLLAPSE — rank-2 cluster"
        if rank_90 <= 3 and off_diag.mean() > 0.6 else
        "USER-SPECIFIC ✓"
        if off_diag.mean() < 0.4 else
        "PARTIAL — some user signal"
    )

    print(f"\n=== {label} (n={len(vectors)}) ===")
    print(f"  cosine off-diag: mean={off_diag.mean():.3f} std={off_diag.std():.3f} "
          f"median={np.median(off_diag):.3f}  frac>0.8={ (off_diag>0.8).mean():.2f}")
    print(f"  PCA: rank@90%={rank_90}, rank@95%={rank_95}, var_2pc={var_2pc:.1%}")
    print(f"  verdict: {verdict}")

    return {
        "label": label,
        "n": len(vectors),
        "cosine": {
            "mean": float(off_diag.mean()),
            "std":  float(off_diag.std()),
            "median": float(np.median(off_diag)),
            "frac_above_0.5": float((off_diag > 0.5).mean()),
            "frac_above_0.8": float((off_diag > 0.8).mean()),
        },
        "pca": {
            "rank_90pct": rank_90,
            "rank_95pct": rank_95,
            "var_2pc": var_2pc,
            "top10": pca.explained_variance_ratio_[:10].tolist(),
        },
        "magnitude": {
            "mean_norm": float(np.linalg.norm(vectors, axis=1).mean()),
            "std_norm":  float(np.linalg.norm(vectors, axis=1).std()),
        },
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Downstream evaluation with steering
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_steering_eval(
    model, tokenizer, samples, vectors: np.ndarray, *,
    layer_idx: int, alpha: float,
    chat_kwargs: dict, system_prompt: str,
    max_new_tokens: int,
    label: str,
) -> tuple[list[str], list[str]]:
    preds, refs = [], []
    for i, (s, v) in enumerate(zip(samples, vectors)):
        v_t = torch.from_numpy(v) if alpha != 0 else None
        pred = persona_steered_generate(
            model, tokenizer,
            user_input=s["input_text"],
            persona_vector=v_t, layer_idx=layer_idx, alpha=alpha,
            max_new_tokens=max_new_tokens,
            chat_kwargs=chat_kwargs, system_prompt=system_prompt,
        )
        preds.append(pred)
        refs.append(s["output_text"].strip())
        if (i + 1) % 5 == 0:
            print(f"  {label} {i+1}/{len(samples)}")
    return preds, refs


def primary(metric_name: str, value: dict) -> float:
    if metric_name == "accuracy":
        return value["accuracy"]
    if metric_name == "regression":
        return value["mae"]
    if metric_name == "rouge":
        return value["ROUGE-L"]
    return float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--task", default="LaMP-2")
    ap.add_argument("--layer_idx", type=int, default=13)
    ap.add_argument("--n_users", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="results/positive_control")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    info = task_info(args.task)
    metric = info["metric"]
    chat_kwargs = chat_kwargs_for(args.model)
    system_prompt = system_prompt_for(args.model)

    print(f"=== Positive control: {args.model} on {args.task} ===")
    print(f"  layer={args.layer_idx} α={args.alpha} n_users={args.n_users}")

    # ---- Load model + tokenizer once
    model, tokenizer = load_model_and_tokenizer(args.model)
    n_layers = len(get_decoder_layers(model))
    if not (0 <= args.layer_idx < n_layers):
        raise ValueError(f"layer_idx {args.layer_idx} out of range [0, {n_layers})")

    # ---- Dataset (already builds template positive/negative prompts).
    # unique_users=True deduplicates by profile hash so 30 means 30 *distinct*
    # users (LaMP-2 has ~5 test items per user; LaMP-7 is already 1:1).
    dataset = LaMPDataset(task=args.task, split="val", n_samples=args.n_users,
                          data_dir=str(ROOT / "data"), unique_users=True)
    samples = list(dataset)
    print(f"  loaded {len(samples)} unique-user samples")

    # ---- Step 1: extract facts via local Qwen
    print("\n[1] Extracting facts via local Qwen...")
    fact_cache = ROOT / args.output_dir / f"cache_facts_{args.task}.json"
    extractor = FactExtractor(model=model, tokenizer=tokenizer, task=args.task)
    samples_enriched = extractor.build_artifacts_for_dataset(
        samples=samples, n_users=args.n_users, cache_path=str(fact_cache),
    )

    # ---- Step 2: template-based vectors
    print(f"\n[2] Template-based vector extraction (layer {args.layer_idx})...")
    template_vectors = extract_vectors(
        model, tokenizer, samples_enriched,
        layer_idx=args.layer_idx,
        positive_key="positive_system_prompts",
        negative_key="negative_system_prompts",
        chat_kwargs=chat_kwargs,
    )

    # ---- Step 3: fact-based vectors
    print(f"\n[3] Fact-based vector extraction (layer {args.layer_idx})...")
    fact_vectors = extract_vectors(
        model, tokenizer, samples_enriched,
        layer_idx=args.layer_idx,
        positive_key="fact_positive_prompts",
        negative_key="fact_negative_prompts",
        chat_kwargs=chat_kwargs,
    )

    # ---- Step 4: geometry
    print("\n[4] Geometry comparison")
    template_geo = compute_geometry(template_vectors, "Template-based")
    fact_geo = compute_geometry(fact_vectors, "Fact-based")

    # ---- Step 5: zero-shot baseline + downstream steering
    print("\n[5] Downstream evaluation")
    zs_preds, zs_refs = run_steering_eval(
        model, tokenizer, samples_enriched, np.zeros_like(template_vectors),
        layer_idx=args.layer_idx, alpha=0.0,
        chat_kwargs=chat_kwargs, system_prompt=system_prompt,
        max_new_tokens=info["max_new_tokens"], label="zs",
    )
    zs_metric = compute_metric(metric, zs_preds, zs_refs)
    zs_p = primary(metric, zs_metric)
    print(f"  zero-shot: {zs_metric}")

    tmpl_preds, tmpl_refs = run_steering_eval(
        model, tokenizer, samples_enriched, template_vectors,
        layer_idx=args.layer_idx, alpha=args.alpha,
        chat_kwargs=chat_kwargs, system_prompt=system_prompt,
        max_new_tokens=info["max_new_tokens"], label="template",
    )
    tmpl_metric = compute_metric(metric, tmpl_preds, tmpl_refs)
    tmpl_p = primary(metric, tmpl_metric)
    print(f"  template steering: {tmpl_metric}")

    fact_preds, fact_refs = run_steering_eval(
        model, tokenizer, samples_enriched, fact_vectors,
        layer_idx=args.layer_idx, alpha=args.alpha,
        chat_kwargs=chat_kwargs, system_prompt=system_prompt,
        max_new_tokens=info["max_new_tokens"], label="fact",
    )
    fact_metric_v = compute_metric(metric, fact_preds, fact_refs)
    fact_p = primary(metric, fact_metric_v)
    print(f"  fact steering: {fact_metric_v}")

    # ---- Step 6: save
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save raw vectors as .npz so plot_comparison.py can re-use them
    npz_path = out_dir / f"vectors_{args.model.split('/')[-1]}_{args.task}.npz"
    np.savez_compressed(npz_path,
                        template=template_vectors, fact=fact_vectors)

    sign = -1 if metric == "regression" else 1   # MAE: lower is better
    payload = {
        "model": args.model,
        "task": args.task,
        "metric": metric,
        "layer_idx": args.layer_idx,
        "n_users": args.n_users,
        "alpha": args.alpha,
        "extraction_method": "local_llm_no_api",
        "lamp_profile_format": "pre_formatted_strings (titles_p6)",
        "geometry": {
            "template": template_geo,
            "fact_based": fact_geo,
            "cosine_delta": fact_geo["cosine"]["mean"] - template_geo["cosine"]["mean"],
            "rank_90_delta": fact_geo["pca"]["rank_90pct"] - template_geo["pca"]["rank_90pct"],
            "hypothesis_supported": (
                fact_geo["cosine"]["mean"] < template_geo["cosine"]["mean"] - 0.15
                and fact_geo["pca"]["rank_90pct"] > template_geo["pca"]["rank_90pct"]
            ),
        },
        "downstream": {
            "zero_shot":         {"value": zs_metric,  "primary": zs_p},
            "template_steering": {"value": tmpl_metric, "primary": tmpl_p,
                                  "delta": sign * (tmpl_p - zs_p)},
            "fact_steering":     {"value": fact_metric_v, "primary": fact_p,
                                  "delta": sign * (fact_p - zs_p)},
        },
        "samples_for_inspection": [
            {"id": s.get("id"), "facts": s["extracted_facts"]}
            for s in samples_enriched[:3]
        ],
        "vectors_npz": str(npz_path.relative_to(ROOT)),
        "timestamp": datetime.now().isoformat(),
    }
    out_path = out_dir / f"comparison_{args.model.split('/')[-1]}_{args.task}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)
    print(f"  template cosine:       {template_geo['cosine']['mean']:.3f}")
    print(f"  fact-based cosine:     {fact_geo['cosine']['mean']:.3f}")
    print(f"  cosine Δ:              {payload['geometry']['cosine_delta']:+.3f}")
    print(f"  template rank@90%:     {template_geo['pca']['rank_90pct']}")
    print(f"  fact-based rank@90%:   {fact_geo['pca']['rank_90pct']}")
    print(f"  hypothesis supported:  {payload['geometry']['hypothesis_supported']}")
    print(f"  ZS / tmpl / fact:      {zs_p:.3f} / {tmpl_p:.3f} / {fact_p:.3f}")
    print(f"  ΔRetrieval:  template={payload['downstream']['template_steering']['delta']:+.3f}  "
          f"fact={payload['downstream']['fact_steering']['delta']:+.3f}")
    print(f"\nSaved {out_path}")
    print(f"Saved {npz_path}")


if __name__ == "__main__":
    main()
