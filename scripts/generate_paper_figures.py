"""Build all paper figures (PDF) and tables (LaTeX) from JSON results.

Outputs:
    figures/fig1_layer_search.pdf       — accuracy vs layer fraction (multi-model)
    figures/fig2_magnitude.pdf          — magnitude ratio histogram (key mech-interp)
    figures/fig3_cosine_sim.pdf         — per-user cosine similarity heatmap
    figures/fig4_main_results.pdf       — bar chart, ZS vs persona × 6 tasks
    figures/fig5_alpha_curve.pdf        — accuracy vs α on LaMP-2 (sweep curve)
    figures/fig6_n_questions.pdf        — accuracy vs n_questions (extraction noise)
    paper/tables/table1_main.tex        — main results table (LaTeX, ACL)
    paper/tables/table2_layer_search.tex — best layer per (model, task)
    paper/tables/table3_geometry.tex    — geometry summary across models

Usage:
    python scripts/generate_paper_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
TABLES = ROOT / "paper" / "tables"

FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def load_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def short(model_name: str) -> str:
    return model_name.split("/")[-1].replace("-Instruct-2501", "")


def primary_value(metric: str, value: dict) -> float:
    if metric == "accuracy":
        return value["accuracy"]
    if metric == "regression":
        return value["mae"]
    if metric == "rouge":
        return value["ROUGE-L"]
    return float("nan")


def primary_label(metric: str) -> str:
    return {"accuracy": "Accuracy", "regression": "MAE", "rouge": "ROUGE-L"}[metric]


# ---------------------------------------------------------------------------
# Fig 1 — layer search curves
# ---------------------------------------------------------------------------


def plot_layer_search():
    files = sorted((RESULTS / "layer_search").glob("layer_search_*.json"))
    if not files:
        print("[fig1] no layer_search results — skipping"); return

    by_task: dict[str, list[dict]] = {}
    for p in files:
        d = load_json(p)
        by_task.setdefault(d["task"], []).append(d)

    n_tasks = len(by_task)
    fig, axes = plt.subplots(1, n_tasks, figsize=(6 * n_tasks, 4), squeeze=False)
    axes = axes[0]

    for ax, (task, runs) in zip(axes, by_task.items()):
        for d in runs:
            metric = d["results"][0]["metric"] if d["results"] else "accuracy"
            xs = [r["layer_fraction"] for r in d["results"]]
            ys = [primary_value(metric, r["value"]) for r in d["results"]]
            ax.plot(xs, ys, marker="o", linewidth=2, label=short(d["model"]))
            best = d["best_layer"]
            ax.axvline(best["layer_fraction"], linestyle="--", alpha=0.4)
            # Zero-shot reference (dashed horizontal line).
            zs_val = primary_value(metric, d["zero_shot"]["value"])
            ax.axhline(zs_val, linestyle=":", color="gray", alpha=0.5,
                       label=f"{short(d['model'])} ZS")
        ax.set_xlabel("Layer (fraction of total depth)")
        ax.set_ylabel(primary_label(d["results"][0]["metric"]))
        ax.set_title(task)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle("Persona-vector quality vs extraction layer", y=1.02, fontsize=13)
    plt.tight_layout()
    out = FIGURES / "fig1_layer_search.pdf"
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"[fig1] {out}")


# ---------------------------------------------------------------------------
# Fig 2 — magnitude ratio  (composite across geometry runs)
# ---------------------------------------------------------------------------


def plot_magnitude():
    files = sorted((RESULTS / "geometry").glob("geometry_*.json"))
    if not files:
        print("[fig2] no geometry results — skipping"); return

    fig, ax = plt.subplots(figsize=(7, 4))
    means, labels = [], []
    for p in files:
        d = load_json(p)
        means.append(d["magnitude"]["mean_magnitude_ratio"])
        labels.append(f"{short(d['model'])}\n{d['task']} L{d['layer_idx']}")

    ypos = np.arange(len(labels))
    ax.barh(ypos, [m * 100 for m in means], color="steelblue", alpha=0.85)
    for i, m in enumerate(means):
        ax.text(m * 100 + 0.3, i, f"{m:.1%}", va="center", fontsize=9)
    ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(r"$\|v_{\mathrm{user}}\| \, / \, \|h_{\mathrm{residual}}\|$ (%)")
    ax.set_title("Persona vector magnitude relative to residual stream\n"
                 "(steering strength is proportional to this ratio)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out = FIGURES / "fig2_magnitude.pdf"
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"[fig2] {out}")


# ---------------------------------------------------------------------------
# Fig 4 — main results bar chart  (uses results/main_table from smoke runs)
# ---------------------------------------------------------------------------


def plot_main_results():
    """Read all baseline_/persona_ JSONs from results/main_table and plot."""
    files = list((RESULTS / "main_table").glob("*.json"))
    if not files:
        print("[fig4] no main_table results — skipping"); return

    rows: dict[tuple[str, str], dict] = {}
    for p in files:
        d = load_json(p)
        if not d:
            continue
        llm = short(d["llm"])
        task = d["task"]
        exp = d.get("experiment", "?")
        metric = d["result"]["metric"]
        v = primary_value(metric, d["result"]["value"])
        rows.setdefault((llm, task), {"metric": metric})[exp] = v

    # tasks order
    tasks = ["LaMP-1", "LaMP-2", "LaMP-3", "LaMP-4", "LaMP-5", "LaMP-7"]
    llms = sorted({k[0] for k in rows})

    fig, axes = plt.subplots(1, len(tasks), figsize=(3 * len(tasks), 4), sharey=False)
    for ax, task in zip(axes, tasks):
        vals_zs, vals_ps, llabels = [], [], []
        for llm in llms:
            r = rows.get((llm, task))
            if not r:
                continue
            llabels.append(llm)
            vals_zs.append(r.get("zero_shot_control", float("nan")))
            vals_ps.append(r.get("persona_steering", float("nan")))
        x = np.arange(len(llabels))
        w = 0.35
        ax.bar(x - w / 2, vals_zs, w, label="Zero-shot", color="lightgray")
        ax.bar(x + w / 2, vals_ps, w, label="Persona α=1", color="steelblue")
        ax.set_xticks(x)
        ax.set_xticklabels([l.replace("Mistral-Small-24B", "Mistral-24B")
                            for l in llabels], rotation=30, fontsize=8)
        ax.set_title(task, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        if task == tasks[0]:
            ax.set_ylabel("Primary metric")
            ax.legend(fontsize=8)
    plt.suptitle("Zero-shot vs persona-steered, smoke n=200", y=1.02)
    plt.tight_layout()
    out = FIGURES / "fig4_main_results.pdf"
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"[fig4] {out}")


# ---------------------------------------------------------------------------
# Fig 5 — α-sweep curve
# ---------------------------------------------------------------------------


def plot_alpha_sweep():
    files = sorted((RESULTS / "alpha_sweep").glob("alpha_sweep_*.json"))
    if not files:
        print("[fig5] no alpha_sweep results — skipping"); return

    fig, ax = plt.subplots(figsize=(7, 4))
    for p in files:
        d = load_json(p)
        metric = d["results"][0]["metric"]
        xs = [r["alpha"] for r in d["results"]]
        ys = [primary_value(metric, r["value"]) for r in d["results"]]
        ax.plot(xs, ys, marker="o", label=f"{short(d['model'])} / {d['task']} L{d['layer_idx']}")
    ax.set_xlabel(r"$\alpha$ (steering scale)")
    ax.set_ylabel(primary_label(metric))
    ax.set_title("Persona steering response curve")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = FIGURES / "fig5_alpha_curve.pdf"
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"[fig5] {out}")


# ---------------------------------------------------------------------------
# Fig 6 — n_questions
# ---------------------------------------------------------------------------


def plot_n_questions():
    files = sorted((RESULTS / "n_questions").glob("n_questions_*.json"))
    if not files:
        print("[fig6] no n_questions results — skipping"); return

    fig, ax = plt.subplots(figsize=(7, 4))
    for p in files:
        d = load_json(p)
        metric = d["results"][0]["metric"]
        xs = [r["n_questions"] for r in d["results"]]
        ys = [primary_value(metric, r["value"]) for r in d["results"]]
        ax.plot(xs, ys, marker="s", label=f"{short(d['model'])} / {d['task']} L{d['layer_idx']}")
    ax.set_xscale("log")
    ax.set_xlabel("Number of extraction questions")
    ax.set_ylabel(primary_label(metric))
    ax.set_title("Persona-vector quality vs extraction noise")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = FIGURES / "fig6_n_questions.pdf"
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"[fig6] {out}")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def make_table_main():
    """Build LaTeX table from results/main_table + results/full_run."""
    smoke_files = list((RESULTS / "main_table").glob("*.json"))
    full_files = list((RESULTS / "full_run").glob("*.json"))

    # smoke rows
    smoke: dict[tuple[str, str], dict] = {}
    for p in smoke_files:
        d = load_json(p)
        if not d:
            continue
        smoke.setdefault((short(d["llm"]), d["task"]), {})[d.get("experiment", "?")] = d

    rows = (
        ("Trained baselines (Flan-T5-XXL frozen, Q-Former trained)", [
            ("BehavioralTwin", {
                "LaMP-1": ("0.567", False), "LaMP-2": ("0.703", False),
                "LaMP-3": ("0.251", False), "LaMP-4": ("0.179", False),
                "LaMP-5": ("0.437", False), "LaMP-7": ("0.403", False),
            }),
        ]),
    )

    # Stub LaTeX — replaced lazily based on available data.
    body_lines: list[str] = []

    def fmt_val(metric, val):
        if val is None:
            return "--"
        return f"{primary_value(metric, val):.3f}"

    backbones = ["Qwen3-8B", "Qwen3-14B", "Mistral-Small-24B"]
    tasks = ["LaMP-1", "LaMP-2", "LaMP-3", "LaMP-4", "LaMP-5", "LaMP-7"]
    metric_per_task = {
        "LaMP-1": "accuracy", "LaMP-2": "accuracy", "LaMP-3": "regression",
        "LaMP-4": "rouge", "LaMP-5": "rouge", "LaMP-7": "rouge",
    }

    body_lines.append(r"\multicolumn{7}{l}{\textit{Trained baseline (Flan-T5-XXL, Q-Former)}} \\")
    body_lines.append(r"BehavioralTwin & 0.567 & 0.703 & 0.251 & 0.179 & 0.437 & 0.403 \\")
    body_lines.append(r"\midrule")

    for bb in backbones:
        body_lines.append(rf"\multicolumn{{7}}{{l}}{{\textit{{Training-free: {bb} (frozen)}}}} \\")
        for label, exp in [("Zero-shot", "zero_shot_control"),
                           (r"Persona ($\alpha{=}1$)", "persona_steering")]:
            cells = []
            for t in tasks:
                d = smoke.get((bb, t), {})
                if exp in d:
                    metric = d[exp]["result"]["metric"]
                    cells.append(f"{primary_value(metric, d[exp]['result']['value']):.3f}")
                else:
                    cells.append("--")
            body_lines.append(rf"{label} & {' & '.join(cells)} \\")
        # delta row
        delta_cells = []
        for t in tasks:
            d = smoke.get((bb, t), {})
            zs = d.get("zero_shot_control"); ps = d.get("persona_steering")
            if zs and ps and zs["result"]["metric"] == ps["result"]["metric"]:
                m = zs["result"]["metric"]
                delta = primary_value(m, ps["result"]["value"]) - primary_value(m, zs["result"]["value"])
                # Sign: for MAE (regression), lower is better, so flip.
                sign = -1 if m == "regression" else 1
                eff = sign * delta
                delta_cells.append(f"{eff:+.3f}")
            else:
                delta_cells.append("--")
        body_lines.append(rf"$\Delta$ & {' & '.join(delta_cells)} \\")
        body_lines.append(r"\midrule")

    latex = (
        "\\begin{table*}[t]\n"
        "\\centering\\small\n"
        "\\begin{tabular}{lcccccc}\n"
        "\\toprule\n"
        "\\textbf{Method} & "
        "\\textbf{LaMP-1} & \\textbf{LaMP-2} & \\textbf{LaMP-3} & "
        "\\textbf{LaMP-4} & \\textbf{LaMP-5} & \\textbf{LaMP-7} \\\\\n"
        " & Acc$\\uparrow$ & Acc$\\uparrow$ & MAE$\\downarrow$ & "
        "R-L$\\uparrow$ & R-L$\\uparrow$ & R-L$\\uparrow$ \\\\\n"
        "\\midrule\n"
        + "\n".join(body_lines) +
        "\n\\bottomrule\n\\end{tabular}\n"
        "\\caption{Main results. $\\Delta$ rows show effect of persona steering on the "
        "primary metric (sign-corrected: positive = improvement). Smoke evaluation "
        "with $n{=}200$.}\n"
        "\\label{tab:main}\n"
        "\\end{table*}\n"
    )
    out = TABLES / "table1_main.tex"
    out.write_text(latex)
    print(f"[tab1] {out}")


def make_table_layer_search():
    files = sorted((RESULTS / "layer_search").glob("layer_search_*.json"))
    if not files:
        print("[tab2] no layer_search results — skipping"); return
    rows = []
    for p in files:
        d = load_json(p)
        b = d["best_layer"]
        metric = b["metric"]
        rows.append((short(d["model"]), d["task"],
                     d["n_layers_total"], b["layer_idx"],
                     b["layer_fraction"], primary_value(metric, b["value"]),
                     primary_value(metric, d["zero_shot"]["value"])))
    body = "\n".join(
        rf"{m} & {t} & {n} & {li} & {lf:.2f} & {best:.3f} & {zs:.3f} & {best - zs:+.3f} \\"
        for (m, t, n, li, lf, best, zs) in rows
    )
    latex = (
        "\\begin{table}[t]\n\\centering\\small\n"
        "\\begin{tabular}{llcccccc}\n\\toprule\n"
        "Model & Task & $L$ & best & frac & metric@best & metric@ZS & $\\Delta$ \\\\\n"
        "\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n"
        "\\caption{Optimal extraction layer per (model, task), and effect over zero-shot.}\n"
        "\\label{tab:layer_search}\n\\end{table}\n"
    )
    out = TABLES / "table2_layer_search.tex"
    out.write_text(latex)
    print(f"[tab2] {out}")


def make_table_geometry():
    files = sorted((RESULTS / "geometry").glob("geometry_*.json"))
    if not files:
        print("[tab3] no geometry results — skipping"); return
    rows = []
    for p in files:
        d = load_json(p)
        rows.append((short(d["model"]), d["task"], d["layer_idx"], d["n_users"],
                     d["cosine_similarity"]["mean_off_diagonal"],
                     d["magnitude"]["mean_magnitude_ratio"],
                     d["pca"]["total_explained_2pc"]))
    body = "\n".join(
        rf"{m} & {t} & {li} & {n} & {cos:.3f} & {mr:.3f} & {pca * 100:.1f}\% \\"
        for (m, t, li, n, cos, mr, pca) in rows
    )
    latex = (
        "\\begin{table}[t]\n\\centering\\small\n"
        "\\begin{tabular}{llccccc}\n\\toprule\n"
        "Model & Task & Layer & $n$ & "
        r"$\overline{\cos}_{\mathrm{off}}$ & "
        r"$\overline{\|v\|/\|h\|}$ & "
        r"PCA 2-PC \\"
        "\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n"
        "\\caption{Per-user persona-vector geometry. Low cosine $\\Rightarrow$ vectors "
        "are distinguishable. Low magnitude ratio $\\Rightarrow$ steering signal is small "
        "relative to residual stream norm.}\n"
        "\\label{tab:geometry}\n\\end{table}\n"
    )
    out = TABLES / "table3_geometry.tex"
    out.write_text(latex)
    print(f"[tab3] {out}")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def main():
    print("Generating figures and tables...")
    plot_layer_search()
    plot_magnitude()
    plot_main_results()
    plot_alpha_sweep()
    plot_n_questions()
    print()
    make_table_main()
    make_table_layer_search()
    make_table_geometry()
    print(f"\nFigures: {FIGURES}")
    print(f"Tables:  {TABLES}")


if __name__ == "__main__":
    main()
