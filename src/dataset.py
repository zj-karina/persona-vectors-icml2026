"""LaMPDataset — minimal dataset wrapper for persona-vector experiments.

Each iteration yields a dict containing the LaMP raw fields plus pre-built
positive/negative system prompts (positive = profile-derived templates,
negative = a fixed set of generic baselines).

The original llm-behavior-fusion `LaMPDataset` is much larger because it
serves the trained Q-Former pipeline; here we keep only what persona-vector
extraction and inference need.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


GENERIC_NEGATIVE_PROMPTS: list[str] = [
    "You are a neutral, generic assistant with no particular preferences.",
    "Answer in a generic style, without imitating any specific author.",
    "You are an impartial assistant. Do not reflect any personal voice.",
    "Respond in a default, unstyled manner.",
    "You are a baseline assistant with no user context.",
]

POSITIVE_TEMPLATE = (
    "You are an author whose past work is exemplified by the following item. "
    "Reproduce that author's preferences, style, and topical focus.\n"
    "Item: {profile_excerpt}"
)


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------


TASKS: dict[str, dict] = {
    "LaMP-1": {"folder": "LaMP_1", "metric": "accuracy",   "max_new_tokens": 3},
    "LaMP-2": {"folder": "LaMP_2", "metric": "accuracy",   "max_new_tokens": 3},
    "LaMP-3": {"folder": "LaMP_3", "metric": "regression", "max_new_tokens": 3},
    "LaMP-4": {"folder": "LaMP_4", "metric": "rouge",      "max_new_tokens": 32},
    "LaMP-5": {"folder": "LaMP_5", "metric": "rouge",      "max_new_tokens": 32},
    "LaMP-7": {"folder": "LaMP_7", "metric": "rouge",      "max_new_tokens": 32},
}


def task_info(task: str) -> dict:
    if task not in TASKS:
        raise ValueError(f"Unknown task '{task}'. Use one of {list(TASKS)}.")
    return TASKS[task]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class LaMPDataset:
    """Iterable LaMP dataset for persona experiments.

    Args:
        task: e.g. "LaMP-2"
        split: "train" or "val" (val maps to dev_titles_p6.json)
        n_samples: cap on number of examples (None = all)
        data_dir: root data folder (expects subfolders LaMP_{1..7}/)
        n_positive, n_negative, excerpt_chars: artifact build params
    """

    def __init__(
        self,
        task: str,
        split: str = "val",
        n_samples: int | None = None,
        data_dir: str = "data",
        n_positive: int = 3,
        n_negative: int = 3,
        excerpt_chars: int = 600,
        unique_users: bool = False,
    ):
        """`unique_users=True` deduplicates by profile hash before truncation —
        critical for LaMP-2 (≈5 test items per user) and LaMP-3 (multi-item users).
        """
        info = task_info(task)
        self.task = task
        self.split = split
        self.metric = info["metric"]
        self.max_new_tokens = info["max_new_tokens"]
        self.n_positive = n_positive
        self.n_negative = n_negative
        self.excerpt_chars = excerpt_chars

        fname = "train_titles_p6.json" if split == "train" else "dev_titles_p6.json"
        path = Path(data_dir) / info["folder"] / fname
        if not path.exists():
            raise FileNotFoundError(path)
        with open(path, "r") as f:
            data = json.load(f)
        if unique_users:
            seen: set[str] = set()
            uniq: list[dict] = []
            import hashlib as _h
            for s in data:
                key = _h.md5(json.dumps(s.get("behavior_profile_text", []),
                                       sort_keys=True).encode()).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                uniq.append(s)
            data = uniq
        if n_samples is not None:
            data = data[:n_samples]
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def _build_positive(self, profile_texts: list[str]) -> list[str]:
        if not profile_texts:
            return []
        out = []
        for i in range(self.n_positive):
            item = profile_texts[i % len(profile_texts)][: self.excerpt_chars]
            out.append(POSITIVE_TEMPLATE.format(profile_excerpt=item))
        return out

    def _build_negative(self) -> list[str]:
        neg = list(GENERIC_NEGATIVE_PROMPTS[: self.n_negative])
        if len(neg) < self.n_negative:
            neg = (neg * ((self.n_negative // len(neg)) + 1))[: self.n_negative]
        return neg

    def __getitem__(self, idx: int) -> dict:
        s = self.data[idx]
        profile = s.get("behavior_profile_text") or []
        return {
            "input_text": s["input_text"],
            "output_text": s["output_text"],
            "behavior_profile_text": profile,
            "positive_system_prompts": self._build_positive(profile),
            "negative_system_prompts": self._build_negative(),
        }

    def __iter__(self) -> Iterator[dict]:
        for i in range(len(self)):
            yield self[i]

    def sample_train_inputs(self, k: int, seed: int = 42) -> list[str]:
        """Helper: pull k inputs from the *train* split for use as extraction
        questions. Independent of self.split.
        """
        info = task_info(self.task)
        path = Path("data") / info["folder"] / "train_titles_p6.json"
        if not path.exists():
            return []
        with open(path) as f:
            tr = json.load(f)
        import random
        rng = random.Random(seed)
        if k >= len(tr):
            return [d["input_text"] for d in tr]
        return [d["input_text"] for d in rng.sample(tr, k)]
