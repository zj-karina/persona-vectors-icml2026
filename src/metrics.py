"""LaMP metrics — accuracy / regression (MAE, RMSE) / ROUGE."""

from __future__ import annotations

import re

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_absolute_error, mean_squared_error,
)
from rouge_score import rouge_scorer


def _safe_float(s, default: float = 0.0) -> float:
    try:
        s = str(s).strip()
        m = re.search(r"-?\d+\.?\d*", s)
        if m:
            v = float(m.group())
            return v if np.isfinite(v) else default
        v = float(s)
        return v if np.isfinite(v) else default
    except (ValueError, AttributeError, TypeError):
        return default


def compute_accuracy(preds, labels) -> dict:
    p, l = np.array(preds), np.array(labels)
    return {
        "accuracy":           float(accuracy_score(l, p)),
        "f1_weighted":        float(f1_score(l, p, average="weighted", zero_division=0)),
        "f1_macro":           float(f1_score(l, p, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(l, p, average="weighted", zero_division=0)),
        "precision_macro":    float(precision_score(l, p, average="macro", zero_division=0)),
        "recall_weighted":    float(recall_score(l, p, average="weighted", zero_division=0)),
        "recall_macro":       float(recall_score(l, p, average="macro", zero_division=0)),
    }


def compute_regression(preds, labels) -> dict:
    p = np.array([_safe_float(x) for x in preds])
    l = np.array([_safe_float(x) for x in labels])
    return {
        "mae":  float(mean_absolute_error(l, p)),
        "rmse": float(np.sqrt(mean_squared_error(l, p))),
    }


def compute_rouge(preds, labels) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = [scorer.score(str(l), str(p)) for p, l in zip(preds, labels)]
    n = max(len(scores), 1)
    return {
        "ROUGE-1": float(sum(s["rouge1"].fmeasure for s in scores) / n),
        "ROUGE-L": float(sum(s["rougeL"].fmeasure for s in scores) / n),
    }


def compute_metric(metric: str, preds, labels) -> dict:
    if metric == "accuracy":
        return compute_accuracy(preds, labels)
    if metric == "regression":
        return compute_regression(preds, labels)
    if metric == "rouge":
        return compute_rouge(preds, labels)
    raise ValueError(f"Unknown metric: {metric}")
