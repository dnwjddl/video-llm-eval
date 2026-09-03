"""Unified item schema + circular (option-rotation) utilities for the blind filter.

An *item* is one multiple-choice question. Options are stored WITHOUT letter
prefixes so we can rotate them freely. Rotation shift s means the option list is
cyclically shifted left by s (options[s:] + options[:s]); the answer index moves
accordingly. A question "passes blind" only if the text-only LLM answers every
rotation correctly, which drives the chance of a lucky pass to (1/n)^k.
"""
from __future__ import annotations

import json
import re
import string
from dataclasses import asdict, dataclass, field
from typing import Iterable, List, Optional

import pandas as pd

LETTERS = string.ascii_uppercase
_PREFIX_RE = re.compile(r"^\s*[\(\[]?([A-Za-z])[\)\].:]\s+")

DEFAULT_POST_PROMPT = "Answer with the option's letter from the given choices directly."


@dataclass
class Item:
    item_id: str          # "<benchmark>:<unique id>"
    benchmark: str
    category: str         # benchmark-author category (task / question_type / area ...)
    question: str
    options: List[str]    # option texts, no letter prefixes
    answer_idx: int       # index into options
    video_path: str = ""  # resolved absolute path ("" if unresolved)
    video_id: str = ""    # benchmark-native video identifier
    subcategory: str = ""
    extra: dict = field(default_factory=dict)

    def n_options(self) -> int:
        return len(self.options)


def strip_letter_prefix(text: str) -> str:
    """'A. foo' / '(B) bar' / 'C: baz' -> 'foo' / 'bar' / 'baz'."""
    return _PREFIX_RE.sub("", str(text), count=1).strip()


def rotation_shifts(n_options: int, k: int) -> List[int]:
    """Distinct cyclic shifts to evaluate (at most n_options)."""
    return list(range(min(max(k, 1), max(n_options, 1))))


def rotate(options: List[str], answer_idx: int, shift: int):
    n = len(options)
    shift = shift % n
    new_options = options[shift:] + options[:shift]
    new_answer = (answer_idx - shift) % n
    return new_options, new_answer


def build_prompt(question: str, options: List[str], post_prompt: str = DEFAULT_POST_PROMPT) -> str:
    lines = [question.strip()]
    for i, opt in enumerate(options):
        lines.append(f"{LETTERS[i]}. {opt}")
    if post_prompt:
        lines.append(post_prompt)
    return "\n".join(lines)


def parse_letter(text: str, n_options: int, options: Optional[List[str]] = None) -> int:
    """Return predicted option index or -1. Mirrors lmms-eval's lenient parsing:
    1) a standalone valid letter, 2) a leading '(X)' / 'X.' prefix, 3) unique option-text match."""
    valid = LETTERS[: max(n_options, 2)]
    s = str(text or "").strip()
    if not s:
        return -1
    up = s.upper()
    m = re.search(r"\b([A-Z])\b", up)
    if m and m.group(1) in valid:
        return valid.index(m.group(1))
    m = re.match(r"^\s*[\(\[]?([A-Z])[\)\].:]?", up)
    if m and m.group(1) in valid:
        return valid.index(m.group(1))
    if options:
        norm = " ".join(s.lower().split())
        hits = [i for i, o in enumerate(options) if o and " ".join(str(o).lower().split()) in norm]
        if len(hits) == 1:
            return hits[0]
    return -1


# ---------- (de)serialisation ----------

def items_to_frame(items: Iterable[Item]) -> pd.DataFrame:
    rows = []
    for it in items:
        d = asdict(it)
        d["options"] = json.dumps(d["options"], ensure_ascii=False)
        d["extra"] = json.dumps(d["extra"], ensure_ascii=False, default=str)
        rows.append(d)
    return pd.DataFrame(rows)


def frame_to_items(df: pd.DataFrame) -> List[Item]:
    out = []
    for r in df.itertuples(index=False):
        out.append(
            Item(
                item_id=r.item_id,
                benchmark=r.benchmark,
                category=r.category,
                question=r.question,
                options=json.loads(r.options),
                answer_idx=int(r.answer_idx),
                video_path=r.video_path or "",
                video_id=r.video_id or "",
                subcategory=getattr(r, "subcategory", "") or "",
                extra=json.loads(r.extra) if isinstance(r.extra, str) else {},
            )
        )
    return out
