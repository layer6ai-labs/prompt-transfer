# Fair Comparison Without Touching the Weights: Asymmetric Prompt Transfer and Leaderboard Bias

---

## 1. Background: how language models are compared today

To find out whether GPT, Claude, Llama or Qwen is stronger, the usual approach is to run each
model on a set of tasks (GSM8K math problems, MMLU knowledge questions, and so on) and rank
them by score. Each question is preceded by an instruction such as "Solve the following math
problem step by step and give the final answer." That instruction is the prompt.

Almost every evaluation framework gives all models the same prompt. The reasoning is natural:
if everyone takes the same exam, the comparison is fair.

But the past two years have produced a lot of evidence that how the prompt is written has a
large effect on the score. Sclar et al. (ICLR 2024) found that merely changing the formatting
of the examples — a different separator, different capitalization — can move the same model's
accuracy by tens of points. Mizrahi et al. (TACL 2024) tested 20 models on 39 tasks and found
that rewording the instruction changes not only the scores but the ranking of the models. In
other words, who comes first on the same exam depends on how the instruction at the top was
written.

This raises a question: if the wording of the prompt changes the ranking, is "everyone uses
the same prompt" actually fair, or does it happen to favor some models?

---

## 2. The problem: is a shared prompt "bad for everyone" or "biased toward some"?

These two situations look similar but mean very different things.

If the shared prompt is simply not well written, then replacing it with any well-written prompt
raises every model's score and leaves the ranking roughly unchanged. That is not a serious
problem.

If the shared prompt is biased toward some models — for instance because it happens to match
the training style of one model family — then replacing it reshuffles the ranking. In that case
the leaderboard reflects not only model ability but also who wrote the prompt.

The most direct way to tell these apart is to let each model use the prompt that suits it best,
then check whether the ranking changes. This is also standard industry practice: before
deploying an application, engineers tune the prompt for the specific model they have chosen.
In recent years a family of automatic prompt-optimization methods (TextGrad, MIPRO, GEPA and
others) can do this tuning from a small number of training examples, and they need nothing
beyond the model's generation API — no access to the model's internals.

There is one basic constraint in this proposal that should be stated up front. **The models people most want to compare — GPT, Claude, Gemini — are closed and available only through an API.**
Any method that modifies model parameters or reads internal states cannot be applied to
them. So if we want to answer "is the leaderboard fair?" and want the answer to cover the models
that are actually being compared, every form of "adaptation" must be training-free and purely
black-box: we may change the text sent to the model, never the model itself. This is both a
limitation and the positioning of this proposal: it is currently the only route that can bring
closed models into a fair comparison.

Under this constraint a second question appears. A prompt tuned for model A, given directly to
model B, often performs worse — sometimes much worse. PromptBridge (Wang et al., 2025) calls
this Model Drifting and gives a vivid example: a prompt optimized to 99% accuracy on GPT-5
drops to 69% on Llama-3.1.

More controlled evidence comes from Zhang et al. (2026, *Prompt Optimization Is a Coin Flip*;
hereafter Coin Flip), who ran six optimizers under equal compute budgets with three repeats on
the same tasks with two executor models. On Claude Haiku 4.5, HelpSteer2 was the only task
where all six optimizers beat zero-shot (best +6.8 points); switching the executor to Amazon
Nova Lite, only one of the six still won on that same task. Task, optimizers, training
questions and scoring were unchanged; only the model changed, and the conclusion "is this task
worth optimizing" flipped. Their explanation is that optimization works when the model "can
but doesn't" produce some output format by default — and that gap is a property of the model,
not the task. Industrial experience agrees: Tripathi et al. (2025) report that an enterprise
search system whose prompts passed every regression test on GPT-4-32k began failing after a
same-family upgrade to GPT-4.1, because the newer model no longer filled in the implicit rules
the old prompts relied on and every requirement had to be spelled out.

All of this shows that "the most suitable prompt" really does differ by model. But for natural-
language prompts nobody has asked the next, finer question: is this drop symmetric? Does a
prompt written for a strong model, given to a weak one, lose as much as a prompt written for a
weak model, given to a strong one? Are the reasons the same? If not, leaderboard bias has an
explainable source. The only setting where this question has been seriously asked is soft
prompts under white-box access (Section 3.3), and the answer there was "clearly asymmetric" —
which is both this proposal's precedent and the hypothesis it will test.

---

## 3. What existing work has done

Around the idea of "adapt each model first, then compare," there are three recent papers. They
fall into two groups: those that need access to the model's internals, and those that only need
the API.

### 3.1 Routes that go inside the model

Zhang, Dominguez-Olmedo and Hardt (ICLR 2026) propose train-before-test: on each benchmark,
give every model the same light fine-tuning (LoRA) before testing. Across 61 open models and
24 benchmarks, they found that under direct evaluation different benchmarks give contradictory
rankings (average agreement only 0.52); after fine-tuning, agreement rises to 0.76 and almost
every pair of benchmarks agrees more. Their explanation: fine-tuning removes differences in how
"prepared" each model already was for each benchmark, and what remains is the model's real
potential.

Erkan et al. (2026) take a lighter route: leave the weights alone and prepend 10 learnable
vectors (a soft prompt) to the input, trained for a few dozen steps. They find that these steps
teach almost nothing except "answer in the required format" — yet that alone raises base-model
scores substantially.

These two papers establish something important: adapting first and then comparing changes the
ranking, and makes it more trustworthy. But both methods need to go inside the model —
fine-tuning needs gradients, soft prompts need the embedding layer — so their experiments can
only be run on open models, which excludes precisely the models the leaderboards care most
about. Under this proposal's constraint they cannot be used as methods, but they provide two
things: evidence that adapt-then-compare is worth doing, and a reference point — if black-box
adaptation can reach their effect, going inside the model is unnecessary.

### 3.2 The API-only route: optimizing the prompt

Sadjoli et al. (ACL 2025 Industry Track; the authors include a HELM maintainer) did exactly
black-box adaptation: they used TextGrad and MIPRO to optimize a separate prompt for each of
5 models (three closed, two open) and compared them on 8 datasets. Rankings changed a lot — the
correlation between rankings before and after optimization averaged only 0.23. On GSM8K, the
top model under the shared prompt was a large closed model; after per-model optimization it
was an 8B open model.

This is the only paper so far to do adapt-then-compare on closed models, but it leaves large
gaps. Each model was tried with only two prompts: the shared one and its own optimized one.
Model B was never tried with model A's prompt, so the paper can only show "optimized beats
unoptimized" and cannot separate the two situations in Section 2. It used only 5 models (with
so few, the ranking correlation is coarse — swapping one pair moves it by ±0.2), no repeated
runs, one optimizer, and the same "critic" model (GPT-4o) to optimize prompts for every model,
which may itself favor the GPT family.

Moreover, the appendix shows that almost every displayed "improvement from optimization" is
actually a formatting issue: the model had the right answer, but its position or format did not
match the scoring script's rule, and the optimized prompt merely made the model put the answer
where the script could find it. This point matters enough to spell out.

**The score changed, but the model did not.** One example from Sadjoli's appendix. With the
original prompt the model answered "...Therefore, Tom has 91 trees left after 10 years." — 91 is
correct. But the scoring rule is "take the last integer in the output," which picked up the 10
at the end of the sentence and marked it wrong. The optimized prompt made the model write
"The final answer is: 91," the script picked up 91, and the answer was marked right. The score
went from wrong to right, but the model's ability did not change at all; what changed was
whether a rule could find the answer.
**In this situation, the effect you measure comes not from the object being measured (the model) but from the measuring instrument (the scoring method).**

Hua et al. (EMNLP 2025) studied this question directly: is prompt sensitivity a flaw in the
model, or a product of the scoring method? Their conclusion is that a large part of it comes
from scoring — extracting answers with fixed rules, or comparing log-likelihoods. When they
switched to having an LLM judge whether the answer is correct, score variation across prompts
shrank substantially and rankings became much more stable. This means that of the ranking
changes Sadjoli et al. report, it is currently unclear how much is the model's and how much is
the scoring script's.

### 3.3 The transfer matrix: drawn twice, analyzed once — and only under white-box access

On the asymmetry question at the end of Section 2, PromptBridge did draw a related figure: on
coding tasks it swapped prompts between models pairwise, producing a matrix in which the
diagonal (a model using its own prompt) is clearly higher than the other cells (using someone
else's). But the figure was only used to argue "therefore we need a transfer tool," after which
the authors went and built the tool. The matrix itself was not analyzed: no check of whether
strong→weak differs from weak→strong, no reading of the ranking implied by each column, and no
separation of score changes that come from the model versus the scoring method.

The one paper that filled in and analyzed such a matrix is Wu, Wu and Mou (ICLR 2024), but
for soft prompts: on the base and large versions of BERT, RoBERTa and ALBERT, they tuned a set
of continuous vectors for each model by gradient descent, then carried it to the other models
through the shared vocabulary. Their source×target matrix has two results that bear directly
on this proposal. First, transfer is clearly asymmetric: base→large is usually usable
(RoBERTa-base → RoBERTa-large 25.1%), while large→base nearly collapses (RoBERTa-large →
RoBERTa-base 3.6%); with RoBERTa-large as the source, no target exceeds 12.5%. Second, they
offer a mechanism: the more expressive the model, the more prompts can solve the same task, so
the one gradient descent finds is one of many and carries more model-specific information
than task semantics — which is why it does not travel. Put differently, an optimized prompt =
task semantics + a model-specific component, and transfer succeeds to the extent the former
dominates. They also found that matching several source models at once transfers better — the
intersection of several sources is closer to pure task semantics.

This work matters to the proposal conceptually, not methodologically: it needs gradients at
the embedding layer, its models are under 340M parameters, and none of it applies to closed
models. Whether its direction of asymmetry (strong sources do not travel) holds for natural-
language prompts is entirely open. Natural-language prompts are human-readable, so the
"model-specific component" will show up in some concrete form — redundant instructions,
missing explicit rules, or wording that only works on one model — and identifying it needs a
different analysis tool.

The closest such tool is Gong and Wen (2026). They split DSPy, TextGrad and GEPA optimization
trajectories into the edits between consecutive steps, classified each edit by type (adding a
meta-instruction, adding chain-of-thought, adding "be concise," and so on), and after
propensity adjustment asked which edit types are associated with gains on which task types.
The result is a clear edit × task interaction: inserting meta-instructions averages −10 points
on math tasks, inserting "be concise" averages −8 on logic tasks. But they only did edit ×
task, never edit × model — their five executor models appear only as covariates in the
propensity model, with no results broken out by model, and the paper never states which model
wrote the edits. Their pipeline can be carried over to this proposal directly, with the
treatment changed from "edit type" to "edit type × transfer direction."

---

## 4. What we will do

The core of this proposal is one matrix and two ways of reading it. Throughout, only the text
sent to the model changes, never the model.

Take N models and optimize one prompt for each, giving N prompts. Then have every model try
every prompt and record the score, producing an N×N table S. S[i][j] is the score of model i
using the prompt optimized for model j. Add one more column S[i][0]: the score of model i using
the original shared prompt.

A three-model illustration:

|          | shared prompt | A's prompt | B's prompt | C's prompt |
|----------|---------------|------------|------------|------------|
| Model A  | 70            | **85**     | 72         | 60         |
| Model B  | 68            | 80         | **83**     | 65         |
| Model C  | 50            | 40         | 55         | **70**     |

**Read by rows, the matrix shows transfer.** A drops from 85 to 72 with B's prompt; B drops
from 83 to 80 with A's prompt — different amounts. That is asymmetry. If A is strong and C is
weak, A's prompt on C (40, worse than the shared prompt) and C's prompt on A (60) may differ in
both size and cause.

**Read by columns, the matrix shows leaderboards.** The column "A's prompt" is "what the
leaderboard would look like if the benchmark happened to adopt A's prompt as the shared
instruction" — A first, B second, C third. The column "C's prompt" puts C first. Every column
is a possible leaderboard, and the difference between columns is how sensitive the leaderboard
is to who wrote the prompt. If all columns rank the models about the same way, the shared prompt
is merely bad, not biased; if every column pushes its own source model upward, it is biased.

The two readings use the same data and complement each other: asymmetry explains where bias
comes from (every prompt carries a "home advantage" for its source model), and bias explains
why asymmetry matters (it distorts model selection).

Because everything goes through the API, closed models can be placed fully into this matrix —
as rows and as columns. Routes that go inside the model cannot do this.

On top of the matrix we add three things, matching the three gaps in Section 3:

1. **Score every output twice** — once with the benchmark's original rule-based script, once
   with a fixed LLM judging whether the answer is correct. The difference between the two is
   the part that comes from the scoring method rather than the model.
2. **Apply Hardt et al.'s criterion**: after per-model prompt optimization, do rankings across
   different benchmarks become more consistent? This asks how far black-box adaptation can
   reproduce the effect of fine-tuning. On open models it can be compared directly with Hardt's
   published LoRA results; on closed models this number has never been computed.
3. **Handle the optimizer's own bias.** Black-box prompt optimization needs a "critic" model to
   propose edits, and for closed targets the critic can only be some API model. We use two
   designs side by side: each model acts as its own critic (symmetric, no third party), and a
   fixed external critic crossed over different families (which directly tests for bias).

---

## 5. Research questions

**Q1: How sensitive is the leaderboard to whose prompt is used?**
We expect clear differences between column rankings, and that each model ranks higher in its
own column than in the others (home advantage). If neither holds, the shared prompt is a quality
issue only, and the bias part of this proposal can be closed.

**Q2: Is transfer asymmetric, and why?**
Write L(i→j) for how much model j loses when using model i's prompt instead of its own. We
expect that when i is stronger than j, L(i→j) exceeds L(j→i): a prompt written for a strong
model hurts a weak one more. This direction agrees with Wu et al.'s soft-prompt finding (the
stronger the source, the more model-specific the prompt), but it has never been tested for
natural-language prompts.

The mechanism matters more. The literature suggests three competing, testable mechanisms, and
they make different predictions about which kinds of edits cause loss in which direction once
the edits in each optimized prompt are classified:

- *Capability gap*: strong→weak loss comes from the prompt asking for things the weak model
  cannot do (multi-step reasoning, complex constraints). Prediction: the damaging edits are the
  complexity-increasing ones.
- *Explicitness mismatch*: Tripathi et al. observe that models with stricter instruction
  following stop filling in implicit rules, so old prompts fail on them because they are **not
  explicit enough**; Sadjoli's appendix has the opposite case — adding one more formatting
  instruction to a model that was already doing well made it answer wrongly, so being **too
  explicit** also hurts. Both concern the same direction (weak→strong or old→new) but attribute
  the loss to under- versus over-specification. Prediction: weak→strong loss concentrates on
  edits that add or remove explicit constraints, with a sign that depends on the target's
  instruction-following style. This also suggests that "strong/weak" and "loose/strict
  instruction following" may be two different axes.
- *Wording specificity*: Coin Flip argues that instruction tuning compresses input phrasing
  into a narrow output distribution, leaving models insensitive to wording. If so, prompts that
  win by wording tricks lose most in transfer, and prompts that win by unlocking an output
  format transfer only when the target has the same "can but doesn't" gap. Prediction: loss
  concentrates on wording and formatting edits — which is exactly the part Q3 sets out to
  separate from the scoring method.

The test borrows Gong and Wen's pipeline directly: text-diff each optimized prompt, assign the
edits to the categories above, and estimate the propensity-adjusted average loss of each edit
type in each transfer direction. One uninteresting explanation must also be ruled out: strong
models are near the ceiling, so any prompt is about the same.

For closed models there is one factor we can record but not control: providers have their own
hidden system prompts behind the API. This may be one source of transfer loss. We cannot remove
it, but we will record "closed vs. open" and check whether asymmetry behaves differently across
the two groups.

**Q3: How much of the effect comes from the scoring method rather than the model?**
Following Hua et al., we expect the Q1 and Q2 effects to shrink under LLM judging but not
vanish. If they vanish, the conclusion of this proposal becomes "the effect of optimizing before
evaluating comes mainly from the scoring method, not the model" — which is also a publishable
result.

**Q4: Can black-box adaptation make rankings consistent across benchmarks?**
Hardt et al. achieved this with fine-tuning. Whether prompts can do it is unknown. All three
outcomes are informative: yes — there is no need to go inside the model, and closed models can
be compared fairly; no — this kind of harmonization requires changing parameters, which is an
important qualification of Hardt's result and marks a principled ceiling on fair comparison of
closed models; partly — prompt adaptation raises consistency without collapsing all benchmarks
into one dimension, which suggests it preserves the distinctions between benchmarks.

**Q5: Does the optimizer favor models from its own family?**
Compare the matrices under the "self-critic" and "fixed external critic" designs; within the
external design, swap critic families and check whether a model's optimized score depends on
the critic's family. This question is specific to the black-box route — fine-tuning has no third
party — and existing prompt-optimization studies generally do not treat it as a variable:
Sadjoli et al. used the same GPT-4o critic for every model; Gong and Wen analyzed five executor
models across three frameworks without ever stating which model wrote the edits; Coin Flip does
not say either. If bias exists, self-critic is the fairer default; if not, an external critic is
cheaper.

---

## 6. A first sketch of the approach

Pick a set of models that must include several closed ones; pick a few benchmarks
that have a training split; use an existing black-box prompt optimizer (TextGrad, MIPRO, GEPA
or similar) to optimize one prompt per model; then have every model try every prompt to obtain
the matrix in Section 4. Score every output in two ways — the benchmark's original rule-based
script and an LLM judge — so that we can see how much of the change comes from the scoring
method rather than the model.

Some open models should be included too, for two reasons: they allow comparisons across sizes
within one family, which is the cleanest setting for looking at asymmetry; and closed models get
updated while open ones do not, so results stay reproducible. If the open set includes a few of
the models Hardt et al. used, Q4 can be compared directly against their published LoRA results
without any fine-tuning on our side.

Coin Flip's negative results set three floors for the design. First, in their variance
decomposition question difficulty accounts for 19–91% of variance, far more than any prompt
effect, so every cell of the matrix must be computed on the same question set, comparisons
must be paired by question, and the test set must be large enough. Second, with only 20
training questions their iterative optimizers overfit (train–test gaps up to 5.6 points) and
half of all runs scored below zero-shot; the training set must be well above that, or the
matrix fills with noise. Third, run their headroom check first: generate 10–20 candidate
prompts per task, and if no model's best candidate beats zero-shot by more than the noise
floor, the task is flat for everyone and adds nothing to the matrix. But tasks must not be
selected for headroom alone — Coin Flip's only success case worked by unlocking JSON output,
which is precisely the scoring-method effect Hua et al. describe, and Q3 needs both
format-sensitive and format-insensitive tasks present.

The matrix can also take one extra column: a prompt optimized jointly for all N models. Wu et
al. found for soft prompts that matching several sources yields a prompt closer to pure task
semantics that transfers better; nobody has tried this for natural-language prompts. If Q1
finds the shared prompt biased, this column is the candidate for a "fairer shared prompt," and
its gap from each model's own prompt is a direct estimate of the size of the model-specific
component.

## References

**Evaluation methodology**
- Sclar, Choi, Tsvetkov, Suhr. *Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design.* ICLR 2024. arXiv:2310.11324
- Mizrahi et al. *State of What Art? A Call for Multi-Prompt LLM Evaluation.* TACL 2024. arXiv:2401.00595
- Polo et al. *Efficient multi-prompt evaluation of LLMs.* NeurIPS 2024. arXiv:2405.17202
- Aali et al. *Structured Prompts Improve Evaluation of Language Models.* 2025. arXiv:2511.20836
- Frick et al. *Prompt-to-Leaderboard.* 2025. arXiv:2502.14855
- Hua, Tang, Gu, Gu, Wong, Qin. *Flaw or Artifact? Rethinking Prompt Sensitivity in Evaluating LLMs.* EMNLP 2025. arXiv:2509.01790
- Biderman et al. *Lessons from the Trenches on Reproducible Evaluation of Language Models.* 2024. arXiv:2405.14782
- Dodge et al. *Show Your Work: Improved Reporting of Experimental Results.* EMNLP 2019. arXiv:1909.03004
- Dominguez-Olmedo, Dorner, Hardt. *Training on the test task confounds evaluation and emergence.* 2024. arXiv:2407.07890

**Adapt before comparing**
- Sadjoli, Siefken, Ghosh, Mai, Dahlmeier. *Optimization before Evaluation: Evaluation with Unoptimised Prompts Can be Misleading.* ACL 2025 Industry Track. arXiv:2604.27637
- Erkan, Boll, Kersting, Deiseroth, Parcalabescu. *Soft-Prompt Tuning for Fair and Efficient LLM Benchmark Evaluation.* 2026. arXiv:2606.12117
- Zhang, Dominguez-Olmedo, Hardt. *Train-before-Test Harmonizes Language Model Rankings.* ICLR 2026. arXiv:2507.05195

**Cross-model transfer**
- Wang, Liu, Wang, Li, Wei, Liu, Bao. *PromptBridge: Cross-Model Prompt Transfer for Large Language Models.* 2025. arXiv:2512.01420
- Wu, Wu, Mou. *Zero-Shot Continuous Prompt Transfer: Generalizing Task Semantics Across Language Models.* ICLR 2024. arXiv:2310.01691
- Tripathi, Nema, Halder, Qiao, Jindal. *Prompt Migration: Stabilizing GenAI Applications with Evolving Large Language Models.* 2025. arXiv:2507.05573

**Optimizer behavior analysis**
- Zhang, Wang, Cui, Qiu, Li, Zhu, He. *Prompt Optimization Is a Coin Flip: Diagnosing When It Helps in Compound AI Systems.* CTB@ICML 2026. arXiv:2604.14585
- Gong, Wen. *Why Prompt Optimization Works, and Why It Sometimes Doesn't: A Causal-Inspired Edit-Level Analysis.* 2026. arXiv:2605.26655

**Optimizers**
- Yuksekgonul et al. *TextGrad: Automatic "Differentiation" via Text.* Nature 2025. arXiv:2406.07496
- Opsahl-Ong et al. *Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs (MIPROv2).* EMNLP 2024. arXiv:2406.11695
- Agrawal et al. *GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning.* ICLR 2026 Oral. arXiv:2507.19457
- Ramnath et al. *A Systematic Survey of Automatic Prompt Optimization Techniques.* EMNLP 2025. arXiv:2502.16923
- Chen et al. *MAPO: Boosting Large Language Model Performance with Model-Adaptive Prompt Optimization.* Findings of EMNLP 2023. arXiv:2407.04118
