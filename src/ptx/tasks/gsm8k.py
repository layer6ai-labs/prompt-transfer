"""GSM8K.

Base prompts and both rule scorers are taken verbatim from
lm-evaluation-harness ``lm_eval/tasks/gsm8k/`` (gsm8k.yaml, gsm8k-cot.yaml).
The harness ships a strict/flexible pair, which is exactly the contrast Q3
needs -- so both rule scorers here have harness provenance rather than one
being ours.
"""

from __future__ import annotations

from ptx.promptspec import Demo, PromptSpec
from ptx.score.base import RegexScorer, Scorer
from ptx.tasks.base import Example, Task, register

# lm-evaluation-harness gsm8k.yaml metric_list.regexes_to_ignore
_IGNORE = (",", r"\$", r"(?s).*#### ", r"\.$")

# lm-evaluation-harness gsm8k-cot.yaml fewshot_config.samples, verbatim.
_COT_DEMOS = [
    (
        "There are 15 trees in the grove. Grove workers will plant trees in the grove "
        "today. After they are done, there will be 21 trees. How many trees did the "
        "grove workers plant today?",
        "There are 15 trees originally. Then there were 21 trees after some more were "
        "planted. So there must have been 21 - 15 = 6. The answer is 6.",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars "
        "are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces "
        "do they have left in total?",
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had "
        "32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 "
        "lollipops. How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. "
        "So he gave Denny 20 - 12 = 8. The answer is 8.",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. "
        "How many toys does he have now?",
        "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then "
        "that is 4 more toys. 5 + 4 = 9. The answer is 9.",
    ),
    (
        "There were nine computers in the server room. Five more computers were "
        "installed each day, from monday to thursday. How many computers are now in "
        "the server room?",
        "There were originally 9 computers. For each of 4 days, 5 more computers were "
        "added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.",
    ),
    (
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he "
        "lost 2 more. How many golf balls did he have at the end of wednesday?",
        "Michael started with 58 golf balls. After losing 23 on tuesday, he had "
        "58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. "
        "The answer is 33.",
    ),
    (
        "Olivia has $23. She bought five bagels for $3 each. How much money does she "
        "have left?",
        "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
        "So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.",
    ),
]

# TextGrad (arXiv:2406.07496) Appendix E.1, the zero-shot system prompt its
# GSM8K prompt-optimization run started from.
_TEXTGRAD_INIT = (
    "You will answer a mathematical reasoning question. Think step by step. "
    "The last line of your response should be of the following format: "
    "'Answer: $VALUE' where VALUE is a numerical value."
)


@register
class GSM8K(Task):
    name = "gsm8k"
    format_sensitive = True
    hf_dataset = "openai/gsm8k"
    hf_config = "main"

    def scaffold(self, ex: Example) -> str:
        return f"Q: {ex.question}\nA:"

    def stop(self) -> tuple[str, ...]:
        # gsm8k-cot.yaml generation_kwargs.until
        return ("Q:", "</s>", "<|im_end|>")

    def base_prompts(self) -> dict[str, PromptSpec]:
        return {
            "harness_cot_8shot": PromptSpec(
                instruction="",
                demos=tuple(
                    Demo(question=f"Q: {q}", answer=f"A: {a}") for q, a in _COT_DEMOS
                ),
                provenance={
                    "source": "lm-evaluation-harness lm_eval/tasks/gsm8k/gsm8k-cot.yaml",
                    "kind": "shared_prompt",
                },
            ),
            "textgrad_zeroshot": PromptSpec(
                instruction=_TEXTGRAD_INIT,
                provenance={
                    "source": "TextGrad arXiv:2406.07496 App. E.1 (initialization)",
                    "kind": "shared_prompt",
                },
            ),
        }

    def scorers(self) -> dict[str, Scorer]:
        return {
            # gsm8k-cot.yaml filter_list[strict-match]
            "strict": RegexScorer(
                scorer_id="gsm8k/strict",
                source="lm-evaluation-harness gsm8k-cot.yaml filter_list[strict-match]",
                pattern=r"The answer is (\-?[0-9\.\,]+).",
                group_select=0,
                regexes_to_ignore=_IGNORE,
            ),
            # gsm8k-cot.yaml filter_list[flexible-extract]: group_select -1 is
            # "the last number anywhere in the response". This is the extractor
            # that produces the Sadjoli-style failure -- a correct answer
            # followed by any other number scores zero.
            "flexible": RegexScorer(
                scorer_id="gsm8k/flexible",
                source="lm-evaluation-harness gsm8k-cot.yaml filter_list[flexible-extract]",
                pattern=r"(-?[$0-9.,]{2,})|(-?[0-9]+)",
                group_select=-1,
                regexes_to_ignore=_IGNORE,
            ),
        }

    @staticmethod
    def parse_gold(raw_answer: str) -> str:
        """gsm8k-cot.yaml doc_to_target: everything after the final '####'."""
        return raw_answer.split("####")[-1].strip()
