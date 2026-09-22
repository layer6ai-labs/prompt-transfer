"""BBH Word Sorting.

From lm-evaluation-harness ``lm_eval/tasks/bbh/cot_zeroshot/`` (word_sorting.yaml,
_cot_zeroshot_template_yaml, utils.py).

This is the Q3 showcase. The gold answer is computable in one line of Python,
and TextGrad still scored it with an LLM judge. Here the same generation can be
scored three ways -- harness strict regex, the harness's own WordSortFilter,
and (later) a judge -- which makes the scoring-method effect visible without
any modelling assumption.
"""

from __future__ import annotations

import re

from ptx.promptspec import PromptSpec
from ptx.score.base import FALLBACK, RegexScorer, ScoreResult, Scorer, harness_find_match, normalize
from ptx.tasks.base import Example, Task, register

# _cot_zeroshot_template_yaml metric_list.regexes_to_ignore
_IGNORE = (r"\.$", ",", r"\\", r"\n", '"')


class WordSortFilterScorer:
    """Transcription of ``WordSortFilter`` from the harness's bbh utils.py.

    It finds every input word inside the response and rebuilds the order,
    de-duplicating so that the *last* mention of each word wins. That makes it
    robust to a model restating the list mid-reasoning.

    The harness does not ``re.escape`` the words when building the alternation
    (``f"\\b{w}\\b"``). That is preserved: a word containing regex metacharacters
    is the benchmark's problem, not ours. A malformed pattern is caught so one
    bad row cannot abort a run, and is recorded in ``meta``.
    """

    scorer_id = "bbh_word_sorting/flexible"
    source = "lm-evaluation-harness bbh/cot_zeroshot/utils.py WordSortFilter"

    def score(self, generation: str, example: Example) -> ScoreResult:
        words = example.question.split("List:")[1].strip().split()
        try:
            regex = re.compile("|".join(f"\\b{w}\\b" for w in words))
        except re.error as exc:
            return ScoreResult(
                self.scorer_id, 0.0, FALLBACK, example.gold, {"regex_error": str(exc)}
            )
        match = regex.findall(generation)
        ordered = reversed(dict.fromkeys(reversed(match)))
        extracted = " ".join(ordered)
        ok = normalize(extracted, _IGNORE) == normalize(example.gold, _IGNORE)
        return ScoreResult(
            scorer_id=self.scorer_id,
            score=1.0 if ok else 0.0,
            extracted=extracted,
            gold=example.gold,
        )


@register
class BBHWordSorting(Task):
    name = "bbh_word_sorting"
    format_sensitive = True
    hf_dataset = "lukaemon/bbh"
    hf_config = "word_sorting"

    def scaffold(self, ex: Example) -> str:
        # word_sorting.yaml doc_to_text
        return f"Q: {ex.question}\nA: Let's think step by step."

    def stop(self) -> tuple[str, ...]:
        return ("</s>", "Q:", "<|im_end|>")

    def base_prompts(self) -> dict[str, PromptSpec]:
        return {
            "harness_cot_zeroshot": PromptSpec(
                # word_sorting.yaml "description"
                instruction="Sort a list of words.",
                provenance={
                    "source": "lm-evaluation-harness bbh/cot_zeroshot/word_sorting.yaml",
                    "kind": "shared_prompt",
                },
            ),
        }

    def scorers(self) -> dict[str, Scorer]:
        return {
            # word_sorting.yaml filter_list[strict-match]. The four lookbehinds
            # only fire on an exact "The answer is " style lead-in, and the
            # trailing (?=.) drops the final character of the match.
            "strict": RegexScorer(
                scorer_id="bbh_word_sorting/strict",
                source="lm-evaluation-harness bbh/cot_zeroshot/word_sorting.yaml "
                "filter_list[strict-match]",
                pattern=(
                    r"((?<=The answer is )(.*)(?=.)|(?<=the answer is )(.*)(?=.)"
                    r"|(?<=The answer: )(.*)(?=.)|(?<=The final answer: )(.*)(?=.))"
                ),
                group_select=0,
                regexes_to_ignore=_IGNORE,
            ),
            "flexible": WordSortFilterScorer(),
        }
