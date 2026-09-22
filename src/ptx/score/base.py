"""Scoring primitives.

Scoring is deliberately not part of a Task. Q3 asks how much of a measured
effect comes from the scoring method rather than the model, and that question
only has a seam to work at if a benchmark's questions and its scorers are
separable. A Task supplies scorers; the pipeline chooses which to apply, and
applies them as a second pass over generations that already exist.

The rule-based scorers below are transcribed from lm-evaluation-harness
*verbatim, including their failure modes*. Do not fix them. A strict extractor
that misses a correct answer is not a bug in our code -- it is the object of
study (see BENCHMARKS.md).
"""

from __future__ import annotations

import re
import sys
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Protocol

FALLBACK = "[invalid]"

_PUNCT_TBL = dict.fromkeys(
    i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P")
)


@dataclass(frozen=True, slots=True)
class ScoreResult:
    """One scorer's verdict on one generation."""

    scorer_id: str
    score: float
    extracted: str
    gold: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def correct(self) -> bool:
        return self.score >= 1.0


class Scorer(Protocol):
    scorer_id: str
    #: Where this scorer came from. Rule scorers must name the harness they
    #: were copied out of; anything we wrote ourselves says so explicitly.
    source: str

    def score(self, generation: str, example: Any) -> ScoreResult: ...


# --------------------------------------------------------------------------
# harness-faithful helpers
# --------------------------------------------------------------------------


def harness_find_match(
    pattern: re.Pattern[str] | str,
    text: str,
    group_select: int = 0,
    fallback: str = FALLBACK,
) -> str:
    """Reproduce ``ExtendedRegexFilter.find_match`` from lm-evaluation-harness.

    Notable behaviours that are load-bearing and must not be "improved":
      * ``group_select`` indexes into ``findall`` results, so ``-1`` means the
        *last* match. That is how the flexible GSM8K extractor ends up taking
        the last number in the response.
      * When a pattern has several groups, ``findall`` yields tuples and the
        first non-empty group wins.
    """
    regex = re.compile(pattern) if isinstance(pattern, str) else pattern
    matches = regex.findall(text)
    if not matches:
        return fallback
    try:
        match = matches[group_select]
    except IndexError:
        return fallback
    if isinstance(match, tuple):
        non_empty = [m for m in match if m]
        if not non_empty:
            return ""
        match = non_empty[0]
    return match.strip()


def normalize(
    s: str,
    regexes_to_ignore: tuple[str, ...] = (),
    ignore_case: bool = True,
    ignore_punctuation: bool = False,
) -> str:
    """Reproduce lm-evaluation-harness ``exact_match`` normalization."""
    for r in regexes_to_ignore:
        s = re.sub(r, "", s)
    if ignore_case:
        s = s.lower()
    if ignore_punctuation:
        s = s.translate(_PUNCT_TBL)
    return s.strip()


@dataclass(frozen=True, slots=True)
class RegexScorer:
    """A rule scorer defined by one extraction regex plus a normalizer."""

    scorer_id: str
    source: str
    pattern: str
    group_select: int = 0
    regexes_to_ignore: tuple[str, ...] = ()
    ignore_case: bool = True
    ignore_punctuation: bool = False

    def score(self, generation: str, example: Any) -> ScoreResult:
        extracted = harness_find_match(self.pattern, generation, self.group_select)
        gold = example.gold
        ok = normalize(
            extracted, self.regexes_to_ignore, self.ignore_case, self.ignore_punctuation
        ) == normalize(
            gold, self.regexes_to_ignore, self.ignore_case, self.ignore_punctuation
        )
        return ScoreResult(
            scorer_id=self.scorer_id,
            score=1.0 if ok else 0.0,
            extracted=extracted,
            gold=gold,
        )
