"""Local-LLM-based fact extraction from LaMP user profiles.

Used for the positive-control experiment: replace the generic template
positive prompt (``You are an author whose past work is...``) with concrete,
user-specific facts elicited from the same Qwen3-8B that we steer.

No external APIs. The fact-extraction LLM is the same model already loaded
for steering, so the cost is one extra `generate()` call per user.

Hypothesis: the rank-2 collapse observed for template-based vectors is a
template confound — it should disappear (or shrink) when positive prompts
contain *concrete user facts* rather than a fixed boilerplate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import torch


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


FACT_EXTRACTION_PROMPT = """Analyze this user's behavioral history and identify what makes them unique.

User history:
{profile_text}

List exactly 5 specific facts about this user that DISTINGUISH them from other users.
Rules:
- Be concrete: name actual topics, genres, patterns, styles from their history.
- Do NOT use generic phrases like "the user likes" or "tends to prefer".
- Each fact must be falsifiable: another user could plausibly NOT have this fact.
- Focus on what is unusual or distinctive, not what is average.

Facts:
1."""


DOMAIN_NEGATIVE_PROMPTS: dict[str, str] = {
    "LaMP-1": (
        "You are assisting a typical academic researcher who reads papers across "
        "various disciplines without strong domain preferences. They follow "
        "mainstream publication venues and general research trends."
    ),
    "LaMP-2": (
        "You are assisting a typical news reader who categorises articles "
        "across mainstream sections without strong topical preferences."
    ),
    "LaMP-3": (
        "You are assisting a typical online shopper who gives average ratings "
        "across product categories without strong brand or style preferences."
    ),
    "LaMP-4": (
        "You are assisting a typical journalist who writes headlines covering "
        "general news in standard AP style without distinctive personal voice."
    ),
    "LaMP-5": (
        "You are assisting a typical academic who writes paper titles in "
        "standard academic style across general interdisciplinary research areas."
    ),
    "LaMP-7": (
        "You are assisting a typical social-media user who paraphrases content "
        "in neutral, standard language without distinctive personal style."
    ),
}


# ---------------------------------------------------------------------------
# Profile formatting
# ---------------------------------------------------------------------------


_TASK_FRAMING: dict[str, str] = {
    "LaMP-1": "Papers this researcher has cited in past work:",
    "LaMP-2": "Articles this user has previously categorised:",
    "LaMP-3": "Reviews this user has written for products:",
    "LaMP-4": "Headlines this writer has produced:",
    "LaMP-5": "Paper titles this scholar has authored:",
    "LaMP-7": "Tweets this user has posted or paraphrased:",
}


def format_profile_from_lamp(
    profile_items: Sequence[str | dict],
    task: str,
    max_items: int = 8,
    max_item_chars: int = 300,
) -> str:
    """Format a LaMP profile (list of strings or dicts) into a numbered text
    block suitable for fact extraction.

    The LaMP `_titles_p6.json` files we use store profiles as pre-formatted
    strings (e.g. ``TITLE: "..."``, ``REVIEW: ...``) — so we just truncate
    and number them. Dict variants (raw LaMP) are flattened to a key:value list.
    """
    framing = _TASK_FRAMING.get(task, "User's history items:")
    lines: list[str] = []
    for i, item in enumerate(profile_items[:max_items]):
        if isinstance(item, dict):
            kv = " | ".join(
                f"{k}: {str(v)[:120]}"
                for k, v in item.items() if k not in ("id", "user_id")
            )
            text = kv[:max_item_chars]
        else:
            text = str(item)[:max_item_chars]
        lines.append(f"  {i+1}. {text}")
    return f"{framing}\n\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class FactExtractor:
    """Use a local LLM to summarise a user's LaMP profile into 5 distinguishing facts."""

    def __init__(self, model, tokenizer, task: str, max_new_tokens: int = 300):
        self.model = model
        self.tokenizer = tokenizer
        self.task = task
        self.max_new_tokens = max_new_tokens
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def extract_facts(self, profile_items: Sequence[str | dict]) -> dict:
        """Run one local generate() call to get user-specific facts."""
        profile_text = format_profile_from_lamp(profile_items, self.task)
        user_msg = FACT_EXTRACTION_PROMPT.format(profile_text=profile_text)

        # Qwen3 thinking-mode disable (paper said: skip <think> for short outputs).
        chat_kwargs = {}
        model_name = getattr(self.tokenizer, "name_or_path", "")
        if "qwen3" in model_name.lower():
            chat_kwargs["enable_thinking"] = False
            sys_msg = "You are a careful analyst. /no_think"
        else:
            sys_msg = "You are a careful analyst."

        messages = [{"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg}]

        if self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **chat_kwargs,
            )
        else:
            prompt = f"{sys_msg}\n\n{user_msg}"

        enc = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **enc,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new_tokens = out[0, enc["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Strip stray <think> blocks just in case.
        if "<think>" in text:
            text = text.split("</think>")[-1].strip()

        # The prompt ends with "1.", so prepend that to capture the first fact.
        if not text.startswith("1."):
            text = "1. " + text.lstrip("1.").lstrip()

        positive_prompt = (
            "You are assisting a specific user. Here are concrete facts about "
            "their preferences and patterns:\n\n"
            f"{text}\n\n"
            "Respond in a way that reflects these specific characteristics."
        )
        negative_prompt = DOMAIN_NEGATIVE_PROMPTS.get(
            self.task,
            "You are a neutral assistant with no particular preferences.",
        )

        return {
            "raw_facts": text,
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "profile_text": profile_text,
        }

    def build_artifacts_for_dataset(
        self,
        samples: Iterable[dict],
        n_users: int = 30,
        cache_path: str | Path | None = None,
    ) -> list[dict]:
        """For each sample, attach `fact_positive_prompts`, `fact_negative_prompts`
        and the raw extracted facts. Caches results keyed by sample id.
        """
        cache: dict[str, dict] = {}
        if cache_path and Path(cache_path).exists():
            with open(cache_path) as f:
                cache = json.load(f)
            print(f"[fact-cache] loaded {len(cache)} entries from {cache_path}")

        out: list[dict] = []
        samples = list(samples)[:n_users]
        for i, sample in enumerate(samples):
            user_id = str(sample.get("id") or sample.get("user_id") or i)

            if user_id in cache:
                facts = cache[user_id]
                tag = "[cached]"
            else:
                profile = sample.get("behavior_profile_text") or sample.get("profile") or []
                facts = self.extract_facts(profile)
                cache[user_id] = facts
                if cache_path:
                    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(cache_path, "w") as f:
                        json.dump(cache, f, indent=2, ensure_ascii=False)
                tag = ""
            print(f"[facts] {i+1}/{len(samples)} user={user_id} {tag}")

            out.append({
                **sample,
                "fact_positive_prompts": [facts["positive_prompt"]],
                "fact_negative_prompts": [facts["negative_prompt"]],
                "extracted_facts": facts["raw_facts"],
            })
        return out
