"""MMLU-Pro.

Scaffold and strict extractor come from lm-evaluation-harness
``lm_eval/tasks/mmlu_pro/`` (_default_template_yaml, utils.py). The zero-shot
instruction is the official one from TIGER-AI-Lab/MMLU-Pro
``cot_prompt_lib/initial_prompt.txt``.

Two dataset facts the adapter has to handle:
  * There is **no train split** -- only test (12,032) and validation (70). Our
    train/dev are carved out of test, disjointly, and that is recorded in the
    split manifest.
  * The option count varies. It is "up to 10" (A-J), not always 10, so the
    letter range is computed per example rather than assumed.
"""

from __future__ import annotations

import re

from ptx.promptspec import PromptSpec, RenderedPrompt
from ptx.score.base import FALLBACK, RegexScorer, ScoreResult, Scorer, harness_find_match, normalize
from ptx.tasks.base import Example, Task, register

CHOICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

# TIGER-AI-Lab/MMLU-Pro cot_prompt_lib/initial_prompt.txt, verbatim.
# '{$}' in the original is the subject; we keep it as an explicit placeholder
# substituted at render time, since subject varies per example while a
# PromptSpec is per-task.
_OFFICIAL = (
    "The following are multiple choice questions (with answers) about {subject}. "
    'Think step by step and then finish your answer with "the answer is (X)" '
    "where X is the correct letter choice."
)


class PermissiveLetterScorer:
    """Loosened answer extraction. **Not from any harness -- we wrote this.**

    The harness ships only ``custom-extract`` for MMLU-Pro, so unlike GSM8K
    there is no published permissive counterpart. This stands in for one: take
    the last bare option letter the response commits to. Labelled as ours so it
    is never mistaken for the benchmark's own rule.
    """

    scorer_id = "mmlu_pro/permissive"
    source = "ptx (not from a harness) - loosened counterpart to custom-extract"

    _PATTERNS = (
        r"answer is \(?([A-J])\)?",
        r"[Aa]nswer:\s*\(?([A-J])\)?",
        r"\(([A-J])\)",
        r"\b([A-J])\b",
    )

    def score(self, generation: str, example: Example) -> ScoreResult:
        extracted = FALLBACK
        for pat in self._PATTERNS:
            got = harness_find_match(pat, generation, group_select=-1)
            if got != FALLBACK:
                extracted = got
                break
        ok = normalize(extracted) == normalize(example.gold)
        return ScoreResult(
            scorer_id=self.scorer_id,
            score=1.0 if ok else 0.0,
            extracted=extracted,
            gold=example.gold,
        )


@register
class MMLUPro(Task):
    name = "mmlu_pro"
    format_sensitive = True
    hf_dataset = "TIGER-Lab/MMLU-Pro"
    hf_config = "default"

    def scaffold(self, ex: Example) -> str:
        """lm-evaluation-harness mmlu_pro/utils.py format_cot_example."""
        options = ex.meta["options"]
        lines = ["Question:", ex.question, "Options:"]
        for i, opt in enumerate(options):
            if i >= len(CHOICES):
                break
            lines.append(f"{CHOICES[i]}. {opt.strip()}")
        lines.append("Answer: Let's think step by step.")
        return "\n".join(lines)

    def stop(self) -> tuple[str, ...]:
        return ("Question:",)

    def render(self, ex: Example, spec: PromptSpec) -> RenderedPrompt:
        """Substitute the per-example subject into the instruction placeholder.

        An optimizer is free to delete the placeholder; that is a legitimate
        edit and the substitution simply becomes a no-op.
        """
        subject = ex.meta.get("category", "")
        if "{subject}" in spec.instruction:
            spec = spec.__class__(
                instruction=spec.instruction.replace("{subject}", subject),
                system=spec.system,
                demos=spec.demos,
                output_format=spec.output_format,
                render_version=spec.render_version,
                provenance=spec.provenance,
            )
        return super().render(ex, spec)

    def base_prompts(self) -> dict[str, PromptSpec]:
        return {
            "official_zeroshot": PromptSpec(
                instruction=_OFFICIAL,
                provenance={
                    "source": "TIGER-AI-Lab/MMLU-Pro cot_prompt_lib/initial_prompt.txt",
                    "kind": "shared_prompt",
                },
            ),
            "harness_zeroshot": PromptSpec(
                instruction="",
                provenance={
                    "source": "lm-evaluation-harness mmlu_pro/_default_template_yaml "
                    "(scaffold only; the harness relies on 5-shot exemplars)",
                    "kind": "shared_prompt",
                },
            ),
        }

    def scorers(self) -> dict[str, Scorer]:
        return {
            # _default_template_yaml filter_list[custom-extract]
            "strict": RegexScorer(
                scorer_id="mmlu_pro/strict",
                source="lm-evaluation-harness mmlu_pro/_default_template_yaml "
                "filter_list[custom-extract]",
                pattern=r"answer is \(?([ABCDEFGHIJ])\)?",
                group_select=0,
                ignore_case=True,
                ignore_punctuation=True,
            ),
            "permissive": PermissiveLetterScorer(),
        }
