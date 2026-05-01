"""Three-panel figure: template-based vs fact-based per-user vector geometry.

Reads the .npz produced by run_positive_control.py and renders:
    A) Cosine-similarity histograms (off-diagonal)
    B) PCA scree plot (cumulative variance)
    C) PCA(2) scatter with arrows linking the same user across both methods

Usage:
    python plot_comparison.py --model Qwen3-8B --task LaMP-2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def off_diag_cos(vecs: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vecs, axis=1, keepdims=True)
    sim = (vecs / np.maximum(n, 1e-8)) @ (vecs / np.maximum(n, 1e-8)).T
    return sim[~np.eye(len(sim), dtype=bool)]


def plot_comparison(template_vectors, fact_vectors, model_name, task, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    tc = off_diag_cos(template_vectors)
    fc = off_diag_cos(fact_vectors)

    # A — cosine histograms
    axes[0].hist(tc, bins=30, alpha=0.6, color="coral",
                 label=f"Template (μ={tc.mean():.2f})", density=True)
    axes[0].hist(fc, bins=30, alpha=0.6, color="steelblue",
                 label=f"Fact-based (μ={fc.mean():.2f})", density=True)
    axes[0].axvline(tc.mean(), color="coral", linestyle="--", lw=2)
    axes[0].axvline(fc.mean(), color="steelblue", linestyle="--", lw=2)
    axes[0].set_xlabel("Pairwise cosine similarity")
    axes[0].set_ylabel("Density")
    axes[0].set_title("A) Inter-user similarity\n(lower = more user-specific)")
    axes[0].legend(fontsize=9)
    axes[0].set_xlim(-0.2, 1.1)

    # B — PCA scree
    pt = PCA().fit(template_vectors)
    pf = PCA().fit(fact_vectors)
    n_show = min(15, len(template_vectors) - 1)
    axes[1].plot(range(1, n_show + 1),
                 np.cumsum(pt.explained_variance_ratio_[:n_show]),
                 "o-", color="coral", lw=2, label="Template")
    axes[1].plot(range(1, n_show + 1),
                 np.cumsum(pf.explained_variance_ratio_[:n_show]),
                 "o-", color="steelblue", lw=2, label="Fact-based")
    axes[1].axhline(0.9, color="gray", linestyle=":", alpha=0.7, label="90%")
    axes[1].set_xlabel("Components")
    axes[1].set_ylabel("Cumulative variance")
    axes[1].set_title("B) PCA scree\n(higher rank = more diverse)")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    # C — joint PCA(2) scatter
    p2 = PCA(n_components=2).fit(np.vstack([template_vectors, fact_vectors]))
    t2 = p2.transform(template_vectors)
    f2 = p2.transform(fact_vectors)
    axes[2].scatter(t2[:, 0], t2[:, 1], c="coral", alpha=0.6, s=60,
                    label="Template", marker="o")
    axes[2].scatter(f2[:, 0], f2[:, 1], c="steelblue", alpha=0.6, s=60,
                    label="Fact-based", marker="^")
    for i in range(min(8, len(template_vectors))):
        axes[2].annotate("", xy=(f2[i, 0], f2[i, 1]),
                        xytext=(t2[i, 0], t2[i, 1]),
                        arrowprops=dict(arrowstyle="->", color="gray", alpha=0.4))
    ev = p2.explained_variance_ratio_
    axes[2].set_xlabel(f"PC1 ({ev[0]:.1%})")
    axes[2].set_ylabel(f"PC2 ({ev[1]:.1%})")
    axes[2].set_title("C) Joint PCA space\n(arrows: same user, two methods)")
    axes[2].legend(fontsize=9)

    plt.suptitle(f"Template vs Fact-Based Persona Vectors — {model_name}, {task}",
                 fontsize=12)
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-8B")
    ap.add_argument("--task", default="LaMP-2")
    ap.add_argument("--vectors_npz", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.vectors_npz is None:
        args.vectors_npz = str(ROOT / "results" / "positive_control"
                                / f"vectors_{args.model}_{args.task}.npz")
    if args.out is None:
        args.out = str(ROOT / "figures"
                        / f"fig_positive_control_{args.model}_{args.task}.pdf")

    data = np.load(args.vectors_npz)
    plot_comparison(data["template"], data["fact"], args.model, args.task, args.out)


if __name__ == "__main__":
    main()
