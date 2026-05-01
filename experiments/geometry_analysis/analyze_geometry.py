"""Geometric analysis of per-user persona vectors.

Three core analyses (the mech-interp meat of the paper):
    1. Cosine-similarity matrix between extracted user vectors.
       Low off-diagonal mean ⇒ vectors are distinguishable.
    2. Magnitude ratio:  ‖v_user‖ / ‖h_residual‖  at the same layer.
       Tiny ratio ⇒ steering signal is too small to flip argmax tokens
       (this is the explanation for the ~neutral steering effect).
    3. PCA(2) projection — visualise the user-vector cloud.

Usage:
    python analyze_geometry.py --model Qwen/Qwen3-8B --task LaMP-2 \\
        --layer_idx 16 --n_users 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import (
    LaMPDataset, PersonaVectors, get_decoder_layers,
    load_model_and_tokenizer, chat_kwargs_for,
)


@torch.no_grad()
def extract_user_vectors_and_norms(
    model, tokenizer, dataset, *,
    layer_idx: int,
    chat_kwargs: dict,
    extraction_questions: list[str],
    n_users: int,
) -> dict:
    pv = PersonaVectors(
        model=model, tokenizer=tokenizer, layer_idx=layer_idx,
        max_new_tokens=50, chat_template_kwargs=chat_kwargs,
    )

    vectors: list[np.ndarray] = []
    residual_norms: list[float] = []
    user_indices: list[int] = []

    t0 = time.time()
    for i, sample in enumerate(dataset):
        if i >= n_users:
            break
        try:
            v = pv.extract(
                positive_prompts=sample["positive_system_prompts"],
                negative_prompts=sample["negative_system_prompts"],
                extraction_questions=extraction_questions,
            )
        except RuntimeError as e:
            print(f"  user {i}: extract failed ({e}) — skipping")
            continue

        # Residual norm at the same layer for the user's input prompt.
        enc = tokenizer(sample["input_text"], return_tensors="pt", truncation=True,
                        max_length=1024).to(next(model.parameters()).device)
        out = model(**enc, output_hidden_states=True, use_cache=False)
        h_last = out.hidden_states[layer_idx + 1][0, -1, :].float().cpu()
        residual_norms.append(float(h_last.norm().item()))
        vectors.append(v.cpu().float().numpy())
        user_indices.append(i)

        if (i + 1) % 10 == 0:
            print(f"  extracted {i+1}/{n_users}   ({time.time()-t0:.0f}s)")

    return {
        "vectors": np.stack(vectors, axis=0),  # [N, hidden]
        "residual_norms": np.array(residual_norms),
        "user_indices": user_indices,
    }


def analyze(data: dict, *, model_name: str, task: str, layer_idx: int,
            output_dir: Path, figures_dir: Path) -> dict:
    vectors = data["vectors"]
    residual_norms = data["residual_norms"]
    n = len(vectors)
    model_short = model_name.split("/")[-1]

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cosine similarity matrix
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normed = vectors / np.maximum(norms, 1e-8)
    cos = normed @ normed.T
    off_diag = cos[~np.eye(n, dtype=bool)]
    cos_stats = {
        "mean_off_diagonal": float(off_diag.mean()),
        "std_off_diagonal":  float(off_diag.std()),
        "median_off_diagonal": float(np.median(off_diag)),
        "frac_above_0.5":    float((off_diag > 0.5).mean()),
        "frac_above_0.8":    float((off_diag > 0.8).mean()),
        "interpretation": (
            "Distinct" if off_diag.mean() < 0.3 else
            ("Moderately similar" if off_diag.mean() < 0.6 else "Highly redundant")
        ),
    }

    fig, ax = plt.subplots(figsize=(7, 6))
    show = cos[: min(50, n), : min(50, n)]
    im = ax.imshow(show, cmap="RdYlBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="cos(v_i, v_j)")
    ax.set_title(f"Per-user persona-vector cosine similarity\n"
                 f"{model_short}, {task}, layer {layer_idx}")
    ax.set_xlabel("User index"); ax.set_ylabel("User index")
    plt.tight_layout()
    plt.savefig(figures_dir / f"cosine_sim_{model_short}_{task}.pdf", bbox_inches="tight")
    plt.close()

    # 2. Magnitude analysis
    vector_norms = np.linalg.norm(vectors, axis=1)
    ratio = vector_norms / np.maximum(residual_norms, 1e-8)
    mag_stats = {
        "mean_vector_norm":    float(vector_norms.mean()),
        "std_vector_norm":     float(vector_norms.std()),
        "mean_residual_norm":  float(residual_norms.mean()),
        "std_residual_norm":   float(residual_norms.std()),
        "mean_magnitude_ratio": float(ratio.mean()),
        "std_magnitude_ratio":  float(ratio.std()),
        "median_magnitude_ratio": float(np.median(ratio)),
        "interpretation": (
            f"|v|/|h| ≈ {ratio.mean():.1%}; "
            + ("too weak for reliable steering" if ratio.mean() < 0.10
               else "sufficient for steering")
        ),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(vector_norms, bins=20, color="steelblue", alpha=0.7,
                 label=r"$\|v_{\mathrm{user}}\|$")
    axes[0].hist(residual_norms, bins=20, color="coral", alpha=0.7,
                 label=r"$\|h_{\mathrm{residual}}\|$")
    axes[0].set_xlabel("L2 norm"); axes[0].set_ylabel("Count"); axes[0].legend()
    axes[0].set_title("Vector magnitude distributions")

    axes[1].hist(ratio * 100, bins=20, color="purple", alpha=0.7)
    axes[1].axvline(ratio.mean() * 100, color="red", linestyle="--",
                    label=f"mean={ratio.mean():.1%}")
    axes[1].set_xlabel(r"$\|v_{\mathrm{user}}\| / \|h_{\mathrm{residual}}\|$ (%)")
    axes[1].set_ylabel("Count"); axes[1].legend()
    axes[1].set_title("Relative magnitude (steering strength proxy)")

    plt.suptitle(f"{model_short}, {task}, layer {layer_idx}", y=1.02)
    plt.tight_layout()
    plt.savefig(figures_dir / f"magnitude_{model_short}_{task}.pdf", bbox_inches="tight")
    plt.close()

    # 3. PCA(2)
    pca = PCA(n_components=2)
    z = pca.fit_transform(vectors)
    explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(z[:, 0], z[:, 1], c=range(n), cmap="viridis", alpha=0.7, s=24)
    plt.colorbar(sc, ax=ax, label="user index")
    ax.set_xlabel(f"PC1 ({explained[0]:.1%})")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%})")
    ax.set_title(f"PCA of per-user persona vectors\n{model_short}, {task}, layer {layer_idx}")
    plt.tight_layout()
    plt.savefig(figures_dir / f"pca_{model_short}_{task}.pdf", bbox_inches="tight")
    plt.close()

    geom = {
        "model": model_name, "task": task, "layer_idx": layer_idx, "n_users": n,
        "cosine_similarity": cos_stats,
        "magnitude": mag_stats,
        "pca": {
            "explained_variance_ratio": explained.tolist(),
            "total_explained_2pc": float(sum(explained[:2])),
        },
    }
    with open(output_dir / f"geometry_{model_short}_{task}.json", "w") as f:
        json.dump(geom, f, indent=2)

    print("\n=== Geometry summary ===")
    print(f"  cosine off-diagonal: mean={cos_stats['mean_off_diagonal']:.3f} "
          f"std={cos_stats['std_off_diagonal']:.3f}  ({cos_stats['interpretation']})")
    print(f"  magnitude ratio:     mean={mag_stats['mean_magnitude_ratio']:.3f} "
          f"std={mag_stats['std_magnitude_ratio']:.3f}  ({mag_stats['interpretation']})")
    print(f"  PCA 2-pc variance:   {geom['pca']['total_explained_2pc']:.1%}")
    return geom


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--task", default="LaMP-2")
    ap.add_argument("--layer_idx", type=int, default=16,
                    help="If omitted, falls back to default; ideally pass best layer "
                         "from results/layer_search.")
    ap.add_argument("--n_users", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="results/geometry")
    ap.add_argument("--figures_dir", default="figures")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args.model)
    chat_kwargs = chat_kwargs_for(args.model)
    n_layers = len(get_decoder_layers(model))
    if not (0 <= args.layer_idx < n_layers):
        raise ValueError(f"layer_idx {args.layer_idx} out of range [0, {n_layers})")

    dataset = LaMPDataset(task=args.task, split="val", n_samples=args.n_users,
                          data_dir=str(ROOT / "data"))
    extraction_questions = dataset.sample_train_inputs(k=1, seed=args.seed)

    data = extract_user_vectors_and_norms(
        model, tokenizer, dataset,
        layer_idx=args.layer_idx, chat_kwargs=chat_kwargs,
        extraction_questions=extraction_questions, n_users=args.n_users,
    )

    analyze(
        data,
        model_name=args.model, task=args.task, layer_idx=args.layer_idx,
        output_dir=ROOT / args.output_dir,
        figures_dir=ROOT / args.figures_dir,
    )


if __name__ == "__main__":
    main()
