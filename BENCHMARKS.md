# Benchmarks from the three optimizer papers

Sourced from the papers: GEPA (arXiv:2507.19457, App. E.1), MIPROv2 (arXiv:2406.11695,
Table 1 + §5.1), TextGrad (arXiv:2406.07496, §Prompt optimization).

**Tier** = our build priority. **Difficulty** = implementation effort (1 trivial - 5 very
hard). They are different axes: AIME is easy to implement but useless to us; IFBench is
fiddly to implement but the most valuable task in the list.

---

## Full table

**Samples** = what is actually in the dataset, split by split. Where a paper re-split the
data, that is shown separately. Our requirement (DESIGN.md 6.5) is **>=150 train** and
**400-500 test**; the last column says whether the dataset can supply that.

| Benchmark | Samples in the dataset | Paper's split | Big enough for us? | Tier | Diff. | Example input | Example gold | Evaluation |
|---|---|---|---|---|---|---|---|---|
| **GSM8K** <br><sub>TextGrad · `openai/gsm8k` (main)</sub> | train **7,473**<br>test **1,319** | DSPy splits | **yes** | **1** | 1 | `Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?` | `Natalia sold 48/2 = <<48/2=24>>24 clips in May.`<br>`Natalia sold 48+24 = <<48+24=72>>72 clips altogether...`<br>`#### 72` → parsed **72** | strict: last integer == gold · permissive: gold appears anywhere · judge |
| **MMLU-Pro** <br><sub>not in the 3 papers · `TIGER-Lab/MMLU-Pro`</sub> | test **12,032**<br>validation **70**<br>**no train split** | — | **yes**, but we must carve train out of test ourselves | **1** | 2 | `Typical advertising regulatory bodies suggest, for example that adverts must not: encourage _____, cause unnecessary _____ or _____, and must not cause _____ offence.` + `options: [9 strings]` | `answer: "I"` (`answer_index: 8`) | strict: regex `answer is \(([A-J])\)` · permissive: last standalone `(A)-(J)` · judge |
| **BBH Word Sorting** <br><sub>TextGrad · `lukaemon/bbh` (word_sorting)</sub> | test **250** (only split) | 250 re-split into 50 / 100 / 100 | **no** — 250 total caps test at ~100-250 | **1** | 2 | `Sort the following words alphabetically: List: thrill splutter panicking scorch same dot prod obstetric malton onus ...` | `barn damp delmarva dot drumhead embezzle entirety greene guru it&t malton obstetric ...` | strict: exact string match · permissive: token-sequence compare · judge — **TextGrad's actual choice** |
| **BBH Object Counting** <br><sub>TextGrad · `lukaemon/bbh` (object_counting)</sub> | test **250** (only split) | 250 re-split into 50 / 100 / 100 | **no** | 2 | 1 | `I have a flute, a piano, a trombone, four stoves, a violin, an accordion, a clarinet, a drum, two lamps, and a trumpet. How many musical instruments do I have?` | `8` | strict: last integer == gold · permissive · judge optional |
| **IFBench** <br><sub>GEPA · `allenai/IFBench_test` + `allenai/IF_multi_constraints_upto5`</sub> | test **300**<br>train pool **95,373** | 150 / 300 / 294 | train **yes**, test **marginal** (300) | **2** | 3 | `What is the female equivalent to chivalry? Include keyword meridian once in your response, keyword gossamer twice, keyword eclipse three times, ...` | **not a string** — `instruction_id_list: ['count:keywords_multiple']` + `kwargs: [{keyword1:'meridian', keyword2:'gossamer', ...}]` | strict: all checkers pass · permissive: fraction passed · **no judge needed — the Q3 control** |
| **ScoNe** <br><sub>MIPRO · `tasksource/scone`</sub> | train **5,010**<br>test **1,000** | 500 / 500 / full | **yes** | 2 | 2 | premise + hypothesis with nested negation | binary entailment label | strict: exact label match · permissive · judge optional |
| **LiveBench-Math** <br><sub>GEPA · `livebench/math`</sub> | test **368** (only split) | 368 split 3 ways (~122 each) | **no** | 2 | 3 | math question, refreshed monthly (contamination-resistant) | numeric / expression | strict: programmatic · permissive |
| **AIME-2025** <br><sub>GEPA</sub> | **30** test (+90 from 2022-24) | 90 train+val, 30 test x5 reps | **no** — far below the floor | 3 | 2 | AIME competition problem | integer 0-999 | strict: exact integer match |
| **Iris / Iris-Typo** <br><sub>MIPRO</sub> | **150** total | 500/500/2k capped by size | **no** | 3 | 1 | 6 real-valued flower features | species label | accuracy |
| **Heart Disease** <br><sub>MIPRO</sub> | **303** total | as above | **no** | 3 | 2 | 13 clinical features | binary label | accuracy (2-module ensemble) |
| **HotPotQA** <br><sub>GEPA, MIPRO</sub> | ~**113K** QA pairs | GEPA 150/300/300 · MIPRO 500/500/2k | yes, but excluded | 3 | 4 | multi-hop factoid question, fullwiki setting | short answer string | exact match — **needs a Wikipedia retriever, 2 modules / 3 LM calls** |
| **HotPotQA Conditional** <br><sub>MIPRO</sub> | as above | 500 / 500 / 2k | yes, but excluded | 3 | 4 | as above; accepted format varies by answer type (person / date / place) | typed answer | custom format-conditional metric |
| **HoVer** <br><sub>GEPA, MIPRO</sub> | ~**26K** claims | 150/300/300 · 500/500/2k | yes, but excluded | 3 | 5 | claim requiring 3-hop document retrieval | set of gold Wikipedia docs | Recall@21 — **4 modules + Wikipedia index** |
| **PUPA** <br><sub>GEPA</sub> | **443** (111/111/221) | 111 / 111 / 221 | **no** | 3 | 5 | user query containing PII, delegated to an untrusted model | quality + PII-leakage reference | composite score — **needs a 2-model ensemble** |
| **GPQA** <br><sub>TextGrad, *instance* opt only</sub> | **198** (Diamond) | — | **no** | 3 | 2 | graduate-level science MC question | option letter | accuracy — *used for rewriting single answers, not prompt optimization* |
| **MMLU-ML / College Physics** <br><sub>TextGrad, *instance* opt only</sub> | **112** / **102** | — | **no** | 3 | 1 | MMLU subject question | option letter | accuracy — *instance optimization, not prompt optimization* |

### The size problem this table exposes

Most of the optimizer papers' benchmarks are **too small for our design**. Their tasks were
sized for optimizer comparison, where a 100-example test set is acceptable; we need paired
bootstrap CIs across an N x N matrix.

Only **GSM8K, MMLU-Pro and ScoNe** clear 400-500 test examples without touching the
multi-module tasks. At our ~20%-disagreement assumption, standard error on a paired
difference is `sqrt(0.2/n)`:

| n | SE | tasks at this size |
|---|---|---|
| 250 | 2.8 pp | BBH Word Sorting, BBH Object Counting |
| 300 | 2.6 pp | IFBench |
| 368 | 2.3 pp | LiveBench-Math |
| 500 | 2.0 pp | GSM8K, MMLU-Pro, ScoNe |
| 1,000 | 1.4 pp | MMLU-Pro, ScoNe |

So the small tasks are **usable but wide** — fine for the large transfer effects (10-30 pp
in the literature), too wide for the subtle edit-level contrasts in Q2. Plan to run the
edit-level analysis on MMLU-Pro and ScoNe, and keep Word Sorting for the Q3 mechanism
demonstration where the effect is qualitative, not marginal.

---

## What the data actually looks like

Real records, pulled from the HF datasets server.

### MMLU-Pro — `TIGER-Lab/MMLU-Pro`
Columns: `question_id, question, options, answer, answer_index, cot_content, category, src`

```
question_id  70
question     Typical advertising regulatory bodies suggest, for example that adverts
             must not: encourage _____, cause unnecessary _____ or _____, and must
             not cause _____ offence.
options      ['Safe practices, Fear, Jealousy, Trivial',
              'Unsafe practices, Distress, Joy, Trivial',
              ... 9 options total ...
              'Unsafe practices, Distress, Fear, Serious']
answer       'I'
answer_index 8
cot_content  ''                      <- empty on test rows
category     'business'
src          'ori_mmlu-business_ethics'
```

Three things to know:
1. **Option count varies.** It is "up to 10" (A-J), not always 10 — this row has 9. The
   answer-extraction regex must accept a variable letter range.
2. **`cot_content` is empty on test rows** and populated on the 70 validation rows. Those
   70 are the canonical few-shot CoT exemplars, one batch per category.
3. **The canonical answer format comes from `cot_content`**, which ends `"The answer is
   (A)."` That is where the harness regex `answer is \(([A-J])\)` comes from — take it
   from there rather than inventing one.

A validation row, showing the exemplar format:
```
question     The symmetric group $S_n$ has $n!$ elements ... Find the characteristic
             of the ring 2Z.
options      [10 options]
answer       'A'
cot_content  "A: Let's think step by step. A characteristic of a ring is R is $n$ if
              the statement $ka = 0$ for all $a \in 2Z$ implies that $k$ is a multiple
              of $n$. ... Hence $k=0$ and $n=0$. The answer is (A)."
```

### GSM8K — `openai/gsm8k`, config `main`
Columns: `question, answer`
```
question  Natalia sold clips to 48 of her friends in April, and then she sold half as
          many clips in May. How many clips did Natalia sell altogether in April and May?
answer    'Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n
           Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n
           #### 72'
```
Gold is the text after `####`. The `<<48/2=24>>` calculator annotations must be stripped.

### BBH — `lukaemon/bbh`, configs `word_sorting` / `object_counting`
Columns: `input, target`
```
word_sorting
  input   'Sort the following words alphabetically: List: thrill splutter panicking
           scorch same dot prod obstetric malton onus drumhead delmarva barn embezzle
           it&t damp guru subsist entirety greene'
  target  'barn damp delmarva dot drumhead embezzle entirety greene guru it&t malton
           obstetric onus panicking prod same scorch splutter subsist thrill'

object_counting
  input   'I have a flute, a piano, a trombone, four stoves, a violin, an accordion,
           a clarinet, a drum, two lamps, and a trumpet. How many musical instruments
           do I have?'
  target  '8'
```
Note `it&t` in the gold sort order — the tokenizer and comparison must not normalize
punctuation away.

### IFBench — `allenai/IFBench_test`
Columns: `key, prompt, instruction_id_list, kwargs`
```
key                  1
prompt               'What is the female equivalent to chivalry? Include keyword
                      meridian once in your response, keyword gossamer twice in your
                      response, keyword eclipse three times in your response, keyword
                      threshold five times in your response, and keyword cascade seven
                      times in your response.'
instruction_id_list  ['count:keywords_multiple']
kwargs               [{'keyword1': 'meridian', 'keyword2': 'gossamer',
                       'keyword3': 'eclipse', 'keyword4': 'threshold', ...}]
```
**There is no gold answer string at all.** Gold is a list of checker IDs plus their
arguments; scoring runs the named verifier functions over the model's output. That is why
IFBench sits outside the Q3 confound entirely — there is nothing for a judge or a regex to
get wrong.

Train data is a separate dataset, `allenai/IF_multi_constraints_upto5` (95,373 rows). The
test set holds out 58 constraint types never seen in training, which makes it a genuine
generalization split rather than a random partition.

---

## Two results in these papers that change the proposal

**1. GEPA already ran one cross-model transfer cell — and it gained.**
A prompt optimized entirely on Qwen3-8B, evaluated on GPT-4.1-Mini unmodified
("GEPA-Qwen-Opt"): **+9.00 aggregate**, beating every optimizer that tuned *directly* on
GPT-4.1-Mini (MIPROv2 +5.64, TextGrad +6.11, Trace +3.27).

Weak → strong transfer does not merely survive, it wins. Q2 therefore cannot be "prompts
don't transfer"; it must be specifically that **strong → weak** is the lossy direction —
which is what Wu et al. found for soft prompts. One cell of our matrix now has a published
value to check against.

**2. MIPRO found demonstrations beat instructions.**
"Optimizing bootstrapped demonstrations alone yields significantly better performance than
optimizing instructions alone... in all but one case."

This makes the `instruction_only` vs `full_prompt` split (DESIGN.md 6.3) load-bearing. If
demos carry most of the gain, what transfers is mostly demonstrations written in the source
model's own style — a different phenomenon from instruction transfer, plausibly with a
different asymmetry.

**Train-set sizes in the literature:** TextGrad 50 (36 actually seen), GEPA 150, MIPRO 500.
Coin Flip's overfitting failure was at 20. So 150–500 is the established range.

---

## The scoping decision this list forces

**Restrict the main matrix to single-module tasks.**

GEPA and MIPRO optimize *programs*, not prompts. HotPotQA is 2 modules over a Wikipedia
index, HoVer is 4, PUPA needs a trusted/untrusted model ensemble. For those, "model A's
prompt" is really a prompt *set*, transfer means moving the whole set, and the edit
analysis has to diff several prompts at once.

Dropping them removes the entire retrieval-infrastructure lift and keeps `PromptSpec` as
one object. It costs overlap with the published *programs*, not with the published
*datasets*.

---

## Notes per tier

**Tier 1 — build these first.**
GSM8K is the hello-world: one call, exact match, data ready, no infrastructure. Expect E1
to reject it on ceiling (TextGrad's baseline was gpt-3.5 at 72.9%; Qwen3.6-27B will be
above 90%) — that is the gate working, and a documented rejection of a task the field
still reports on is worth a paragraph. MMLU-Pro is the workhorse: large enough for real
claims and clear of ceiling, unlike plain MMLU. Word Sorting is the Q3 showcase — its
ground truth is computable in one line of Python and TextGrad *still* chose an LLM judge.

**Tier 2 — add once the pipeline is proven.**
IFBench is the most valuable task in the list: programmatic constraint checkers mean no
judge anywhere, so any effect found there cannot be a scoring artifact. It also directly
probes instruction-following strictness — Q2's "explicitness mismatch" mechanism, the same
axis `gemma-3-4b-pt` vs `gemma-3-4b-it` isolates. Collapse GEPA's 2-stage system to one
stage for our purposes.

**Tier 3 — skip.**
AIME (30 test questions, far below the power floor), Iris and Heart Disease (150/303, MIPRO
used them as toys), and the multi-module retrieval tasks.

---

## Appendix: worked scorer disagreements

### GSM8K — the Sadjoli failure on example #1
Gold `72`. Model answers correctly: *"...48 + 24 = 72 clips over the **two** months."*

| scorer | result |
|---|---|
| strict — last integer == gold | **fail** (last integer is `2`) |
| permissive | pass |
| judge | pass |

### MMLU-Pro — regex brittleness
Harness regex expects `answer is (H)`; model writes *"the correct choice is (H)."*
→ strict **fail**, permissive pass, judge pass.

### BBH Word Sorting — three scorers, three verdicts
Gold `costume counterpart oven`; model writes *"Sorting alphabetically: costume,
counterpart, oven."*
→ strict **fail** (commas + preamble), permissive pass, judge pass.

That a task with programmatically computable ground truth still disagrees three ways is Q3
in a single example.

---

## Standing rule for this workstream

Take each base prompt and each strict scorer **verbatim from an existing harness**
(lm-eval-harness, or the optimizer paper's own repo) and record which one in the task
config. The paper argues shared prompts are biased; if we author the shared prompt
ourselves, the rebuttal is that we built a straw man. The scorer's bugs are the
measurement — do not fix them.
