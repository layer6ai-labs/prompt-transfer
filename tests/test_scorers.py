"""The scorer disagreements are the object of study, so they are pinned here.

If one of these tests starts failing because an extractor got "fixed", the fix
is the bug: these transcriptions must keep matching lm-evaluation-harness.
"""

from __future__ import annotations

import pytest

from ptx.tasks import get_task
from ptx.tasks.base import Example


def _ex(task, gold, question="", **meta):
    return Example(id="t", question=question, gold=gold, meta=meta)


# --------------------------------------------------------------------- GSM8K


def test_gsm8k_strict_accepts_harness_format():
    s = get_task("gsm8k").scorers()["strict"]
    r = s.score("Natalia sold 48 + 24 = 72. The answer is 72.", _ex("gsm8k", "72"))
    assert r.correct and r.extracted == "72"


def test_gsm8k_strict_rejects_correct_answer_without_the_lead_in():
    """Right answer, wrong shape. The model is not wrong; the rule cannot see it."""
    s = get_task("gsm8k").scorers()["strict"]
    r = s.score("She sold 48 + 24 = 72 clips altogether.", _ex("gsm8k", "72"))
    assert not r.correct
    assert r.extracted == "[invalid]"


def test_gsm8k_flexible_takes_the_last_number_and_so_misreads_a_correct_answer():
    """The Sadjoli failure: a correct answer followed by any other number.

    group_select=-1 means "last match anywhere", so the trailing '10 years'
    wins over the correct 91.
    """
    s = get_task("gsm8k").scorers()["flexible"]
    r = s.score("Therefore, Tom has 91 trees left after 10 years.", _ex("gsm8k", "91"))
    assert not r.correct
    assert r.extracted == "10"


def test_gsm8k_flexible_rescues_what_strict_rejects():
    """The complementary direction: no lead-in, but the answer ends the sentence."""
    strict = get_task("gsm8k").scorers()["strict"]
    flexible = get_task("gsm8k").scorers()["flexible"]
    gen, ex = "She sold 48 + 24 = 72", _ex("gsm8k", "72")
    assert not strict.score(gen, ex).correct
    assert flexible.score(gen, ex).correct


def test_gsm8k_gold_parse_strips_calculator_annotations():
    from ptx.tasks.gsm8k import GSM8K

    raw = (
        "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
        "Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"
    )
    assert GSM8K.parse_gold(raw) == "72"


# ------------------------------------------------------------------ MMLU-Pro


def test_mmlu_pro_strict_accepts_canonical_phrasing():
    s = get_task("mmlu_pro").scorers()["strict"]
    r = s.score("... so the answer is (I).", _ex("mmlu_pro", "I"))
    assert r.correct


def test_mmlu_pro_strict_is_defeated_by_a_synonym():
    """Regex brittleness: 'choice is' instead of 'answer is'."""
    strict = get_task("mmlu_pro").scorers()["strict"]
    permissive = get_task("mmlu_pro").scorers()["permissive"]
    gen, ex = "Therefore the correct choice is (H).", _ex("mmlu_pro", "H")
    assert not strict.score(gen, ex).correct
    assert permissive.score(gen, ex).correct


def test_mmlu_pro_variable_option_count_is_rendered():
    """MMLU-Pro is 'up to 10' options, not always 10."""
    task = get_task("mmlu_pro")
    ex = _ex("mmlu_pro", "I", question="Q?", options=[f"opt{i}" for i in range(9)], category="business")
    text = task.scaffold(ex)
    assert "I. opt8" in text
    assert "J." not in text


def test_mmlu_pro_subject_placeholder_is_substituted_at_render():
    task = get_task("mmlu_pro")
    spec = task.base_prompts()["official_zeroshot"]
    ex = _ex("mmlu_pro", "A", question="Q?", options=["a", "b"], category="business")
    rendered = task.render(ex, spec)
    assert "about business" in rendered.text
    assert "{subject}" not in rendered.text


# ----------------------------------------------------------- BBH word sorting


WS_Q = "Sort the following words alphabetically: List: oven costume counterpart"
WS_GOLD = "costume counterpart oven"


def test_word_sorting_strict_rejects_a_correct_but_prose_answer():
    s = get_task("bbh_word_sorting").scorers()["strict"]
    r = s.score("Sorting alphabetically: costume, counterpart, oven.", _ex("bbh", WS_GOLD, WS_Q))
    assert not r.correct


def test_word_sorting_flexible_recovers_the_same_answer():
    """Same generation, same task, opposite verdict - this is Q3 in one case."""
    s = get_task("bbh_word_sorting").scorers()["flexible"]
    r = s.score("Sorting alphabetically: costume, counterpart, oven.", _ex("bbh", WS_GOLD, WS_Q))
    assert r.correct, r.extracted


def test_word_sorting_flexible_keeps_the_last_mention_of_each_word():
    """The model restates the list before committing; the final order must win."""
    s = get_task("bbh_word_sorting").scorers()["flexible"]
    gen = "We have oven costume counterpart. Sorted: costume counterpart oven"
    r = s.score(gen, _ex("bbh", WS_GOLD, WS_Q))
    assert r.correct, r.extracted


def test_word_sorting_flexible_survives_regex_metacharacters_in_the_data():
    """'it&t' really is in the dataset; the harness does not escape words."""
    q = "Sort the following words alphabetically: List: it&t barn"
    s = get_task("bbh_word_sorting").scorers()["flexible"]
    r = s.score("The answer is barn it&t", _ex("bbh", "barn it&t", q))
    assert r.correct, r.extracted


# ---------------------------------------------------------------- PromptSpec


def test_prompt_id_is_content_addressed_and_ignores_provenance():
    from ptx.promptspec import PromptSpec

    a = PromptSpec(instruction="Think step by step.", provenance={"source": "x"})
    b = PromptSpec(instruction="Think step by step.", provenance={"source": "y"})
    assert a.prompt_id == b.prompt_id


def test_strip_demos_changes_identity():
    from ptx.promptspec import Demo, PromptSpec

    full = PromptSpec(instruction="i", demos=(Demo("q", "a"),))
    assert full.strip_demos().prompt_id != full.prompt_id
    assert full.strip_demos().demos == ()


@pytest.mark.parametrize("name", ["gsm8k", "mmlu_pro", "bbh_word_sorting"])
def test_every_scorer_declares_its_provenance(name):
    for s in get_task(name).scorers().values():
        assert s.source, f"{s.scorer_id} has no source"
