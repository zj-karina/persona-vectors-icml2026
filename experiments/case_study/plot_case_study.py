"""Render the single-user case study into a multi-panel paper figure.

Reads results/case_study/case_<task>_user<idx>_<variant>.json and produces:
    figures/fig_case_study_<task>_user<idx>_<variant>.pdf

Layout (3 horizontal panels):
    A) Stacked bar chart of top-5 next-token probabilities across α.
    B) Probability of the gold answer vs α (line plot).
    C) Probability of the user's "user-specific" candidate vs α — this is
       the token that the steering should boost if the vector encodes user
       identity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to case_*.json")
    ap.add_argument("--out", default=None, help="PDF output path")
    args = ap.parse_args()

    with open(args.input) as f:
        d = json.load(f)

    if args.out is None:
        stem = Path(args.input).stem
        # Walk up from results/case_study/*.json to project root, then figures/
        repo_root = Path(args.input).resolve().parents[2]
        args.out = str(repo_root / "figures" / f"fig_{stem}.pdf")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    alphas = [r["alpha"] for r in d["rows"]]
    gold = d["gold"].lower()

    # Determine union of top-5 tokens across all α — the panel A bars
    top_tokens: list[str] = []
    for r in d["rows"]:
        for t, _ in r["topk"][:5]:
            if t not in top_tokens:
                top_tokens.append(t)
    top_tokens = top_tokens[:8]  # cap

    # Build prob matrix: tokens × α
    M = np.zeros((len(top_tokens), len(alphas)))
    for j, r in enumerate(d["rows"]):
        for t, p in r["topk"]:
            if t in top_tokens:
                i = top_tokens.index(t)
                M[i, j] = p

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel A — stacked bars (one bar per α, segments per token)
    bottoms = np.zeros(len(alphas))
    cmap = plt.get_cmap("tab10")
    for i, t in enumerate(top_tokens):
        axes[0].bar(range(len(alphas)), M[i], bottom=bottoms,
                    label=repr(t), color=cmap(i % 10), alpha=0.85)
        bottoms = bottoms + M[i]
    axes[0].set_xticks(range(len(alphas)))
    axes[0].set_xticklabels([f"α={a}" for a in alphas])
    axes[0].set_ylabel("P(token | prompt)")
    axes[0].set_title("A) Top-token probabilities at answer position")
    axes[0].legend(fontsize=8, loc="upper right", ncol=2)
    axes[0].set_ylim(0, 1.0)

    # Panel B — gold-token probability vs α
    gold_probs = []
    for r in d["rows"]:
        match = next((p for t, p in r["topk"] if t.lower() == gold), 0.0)
        gold_probs.append(match)
    axes[1].plot(alphas, gold_probs, marker="o", color="seagreen", lw=2)
    axes[1].axhline(gold_probs[0], color="gray", linestyle=":", alpha=0.6,
                    label="zero-shot")
    axes[1].set_xlabel("α (steering scale)")
    axes[1].set_ylabel(f"P(gold = {gold!r})")
    axes[1].set_title("B) Gold-answer probability vs α")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=9)

    # Panel C — pred match (categorical)
    matches = [int(r["match_gold"]) for r in d["rows"]]
    colors = ["seagreen" if m else "lightcoral" for m in matches]
    axes[2].bar(range(len(alphas)), [1] * len(alphas), color=colors, alpha=0.7)
    for j, r in enumerate(d["rows"]):
        axes[2].text(j, 0.5, r["prediction"][:20],
                    ha="center", va="center", fontsize=10,
                    rotation=90 if len(r["prediction"]) > 8 else 0)
    axes[2].set_xticks(range(len(alphas)))
    axes[2].set_xticklabels([f"α={a}" for a in alphas])
    axes[2].set_yticks([])
    axes[2].set_title(f"C) Greedy prediction at α (gold = {gold!r})")

    title = (f"Case study: {Path(args.input).stem}\n"
             f"layer {d['layer_idx']}, |v|={d['vector_norm']:.2f}, "
             f"variant={d['variant']}")
    plt.suptitle(title, fontsize=11, y=1.04)
    plt.tight_layout()
    plt.savefig(args.out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
