# Model registry and GPU layouts for the prompt-transfer pod.
# Sourced by setup.sh and serve.sh; runs on the pod, not the login VM.

# Everything on the pod lives here. The pod disk is ephemeral (no PVC), so a
# restarted pod needs setup.sh again.
PTX_HOME="${PTX_HOME:-${HOME}/ptx}"
SGLANG_VENV="${PTX_HOME}/.venv-sglang"
SGLANG_VERSION="0.5.18"

# Served name -> HF repo, pinned revision, port. Revisions match the copies
# on /layer6share, so pod and share serve identical weights.
declare -A MODEL_REPO=(
  [Qwen3.6-27B]=Qwen/Qwen3.6-27B
  [Qwen3.5-4B]=Qwen/Qwen3.5-4B
  [gemma-4-31B-it]=google/gemma-4-31B-it
  [gemma-4-E4B-it]=google/gemma-4-E4B-it
  [Qwen3-8B]=Qwen/Qwen3-8B
  [gemma-3-27b-it]=google/gemma-3-27b-it
  [gemma-3-12b-it]=google/gemma-3-12b-it
  [gemma-3-4b-it]=google/gemma-3-4b-it
  [gemma-3-1b-it]=google/gemma-3-1b-it
  [Qwen3-32B]=Qwen/Qwen3-32B
  [Qwen3-1.7B]=Qwen/Qwen3-1.7B
  [gpt-oss-20b]=openai/gpt-oss-20b
  [gpt-oss-120b]=openai/gpt-oss-120b
  # Llama 3.1 (2026-10-08): no access to Meta's gated repos, so serving/fetch_llama31.sh
  # puts them on the pod (setup.sh cannot fetch the 8B): the 8B is Meta's bf16 weights
  # (sha256-checked shards from a mirror + Meta's own config/tokenizer files), the 70B
  # RedHat's FP8 checkpoint with Meta's tokenizer files.
  [Llama-3.1-8B-Instruct]=meta-llama/Llama-3.1-8B-Instruct
  [Llama-3.1-70B-Instruct-FP8]=RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8
)
declare -A MODEL_REV=(
  [Qwen3.6-27B]=6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
  [Qwen3.5-4B]=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
  [gemma-4-31B-it]=842da3794eaa0b77d5f08bae87a17459d91ff475
  [gemma-4-E4B-it]=ee0ef6023621cff504d758262d4e04895a5af4a2
  [Qwen3-8B]=b968826d9c46dd6066d109eabc6255188de91218
  [gemma-3-27b-it]=005ad3404e59d6023443cb575daa05336842228a
  [gemma-3-12b-it]=96b6f1eccf38110c56df3a15bffe176da04bfd80
  [gemma-3-4b-it]=093f9f388b31de276ce2de164bdc2081324b9767
  [gemma-3-1b-it]=dcc83ea841ab6100d6b47a070329e1ba4cf78752
  [Qwen3-32B]=9216db5781bf21249d130ec9da846c4624c16137
  [Qwen3-1.7B]=70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
  [gpt-oss-20b]=6cee5e81ee83917806bbde320786a8fb61efebee
  [gpt-oss-120b]=b5c939de8f754692c1647ca79fbf85e8c1e70f8a
  [Llama-3.1-8B-Instruct]=0e9e39f249a16976918f6564b8830bc894c89659
  [Llama-3.1-70B-Instruct-FP8]=f8d0de9953e442c2fd4da32a1d4da8afb8c9452e
)
declare -A MODEL_PORT=(
  [Qwen3.6-27B]=8001
  [Qwen3.5-4B]=8002
  [gemma-4-31B-it]=8003
  [gemma-4-E4B-it]=8004
  [Qwen3-8B]=8005
  [gemma-3-27b-it]=8006
  [gemma-3-12b-it]=8007
  [gemma-3-4b-it]=8008
  [gemma-3-1b-it]=8009
  [Qwen3-32B]=8010
  [Qwen3-1.7B]=8011
  [gpt-oss-20b]=8012
  [gpt-oss-120b]=8013
  [Llama-3.1-8B-Instruct]=8014
  [Llama-3.1-70B-Instruct-FP8]=8015
)

# Repo files not to download (setup.sh --exclude). gpt-oss repos also carry
# the original checkpoint and a Metal build, each as large as the weights.
declare -A MODEL_EXCLUDE=(
  [gpt-oss-20b]="original/* metal/*"
  [gpt-oss-120b]="original/* metal/*"
)

# Prefill chunk size per model; unset means CHUNKED_PREFILL (serve.sh). -1 never
# cuts a prompt. Qwen3.5/3.6 run on fa3, where the chunk budget is shared by the
# requests in flight, so where a long prompt gets cut depends on the traffic,
# and the linear-attention layers then round differently: on 2026-09-25, 35 of
# 300 identical HotpotQA requests came back worded differently between two runs.
# Uncut, a prompt is always prefilled the same way. Gemma 4 has the same
# problem on triton despite the chunk alignment in serve.sh: for hybrid
# sliding-window models SGLang also shortens a prompt's later chunks to the
# free sliding-window pool (schedule_policy add_chunked_req), so on 2026-09-28
# the stress smoke gave 40/48 (31B) and 45/48 (E4B) with 1024-token chunks,
# 48/48 for both uncut. A layout entry can override this (see LAYOUTS).
declare -A MODEL_CHUNKED_PREFILL=(
  [Qwen3.6-27B]=-1
  [Qwen3.5-4B]=-1
  [gemma-4-31B-it]=-1
  [gemma-4-E4B-it]=-1
  [Qwen3-8B]=-1
)

# Extra launch_server flags per model. gpt-oss replies in the harmony format;
# its parser moves the reasoning (analysis channel) into reasoning_content, so
# the reply text is only the final answer. Requests set reasoning_effort "low"
# (gepa-artifact local/series.py): SGLang refuses "none" for harmony.
declare -A MODEL_EXTRA_ARGS=(
  [gpt-oss-20b]="--reasoning-parser gpt-oss"
  [gpt-oss-120b]="--reasoning-parser gpt-oss"
)

# Layout entries are name:gpu:mem_fraction_static:max_running_requests[:chunk],
# started in order. GPU is an index into the job's CUDA_VISIBLE_DEVICES, or
# indices joined by "+" (0+1) for a tensor-parallel server; the optional chunk
# overrides MODEL_CHUNKED_PREFILL for that server. Indices joined by "," (0,1)
# start data-parallel replicas instead, one per GPU behind one port.
#
# SGLang 0.5.18 sizes a server's KV pool from the memory free when it starts
# (pool = free_after_weights - free_before_load * (1 - fraction)), so on a
# shared GPU the big model goes first with a fraction low enough to leave
# room for its partner, and the partner takes most of what remains.
# max_running_requests is capped at 16 in all4 because the Qwen3.5/3.6
# linear-attention state is per request (~147 MiB for 27B, ~49 MiB for 4B).
# all4 fractions are fitted to the first run's logs (2026-09-23): Qwen3.5-4B
# needs ~2.5 GiB of slack for warmup, which leaves gemma-4-31B only ~3.9 GiB.
# Its sliding-window pool reserves ~1.15K tokens per running request plus two
# prefill chunks (~0.8 MiB/token), so it gets 2 concurrent requests here. Use
# strong/weak for bulk runs.
declare -A LAYOUTS=(
  # all four executors resident at once: one big + one small per GPU
  [all4]="Qwen3.6-27B:0:0.72:16 gemma-4-31B-it:1:0.81:2 gemma-4-E4B-it:0:0.87:16 Qwen3.5-4B:1:0.82:16"
  # one strong model per GPU, room for large batches
  [strong]="Qwen3.6-27B:0:0.88:64 gemma-4-31B-it:1:0.88:64"
  # one weak model per GPU
  [weak]="Qwen3.5-4B:0:0.85:64 gemma-4-E4B-it:1:0.85:64"
  # the Qwen pair, one per GPU, for Qwen-only phases. Run with
  # CONTEXT_LENGTH=32768 so GEPA's critic prompts fit (MODELS.md).
  [qwen]="Qwen3.6-27B:0:0.88:32 Qwen3.5-4B:1:0.85:32"
  # the Gemma pair, one per GPU (run both with CONTEXT_LENGTH=32768).
  # gemma-4-31B-it has ~13-15 GiB for KV at 0.93-0.95. Its sliding-window
  # pool holds (1153 * running + 2 * chunk) tokens at 0.78 MiB/token (50 SWA
  # layers x 16 KV heads x 256); the 10 full-attention layers take 0.08
  # MiB/token. Uncut, a whole prompt must fit in the SWA pool, so:
  # gemma: uncut (deterministic), for evaluation. 31B SWA pool 9.2K tokens,
  #   enough for every executor prompt (<5K); longer prompts never run.
  [gemma]="gemma-4-31B-it:0:0.93:8 gemma-4-E4B-it:1:0.85:32"
  # gemma-gepa: 4096-token chunks, for GEPA, whose critic prompts are 15-20K
  #   tokens. Executor prompts (<4096) are still never cut; only longer ones
  #   are, and those may vary with traffic. 31B: SWA 15.1K tokens (11.8 GiB),
  #   full pool ~40K tokens.
  [gemma-gepa]="gemma-4-31B-it:0:0.95:6:4096 gemma-4-E4B-it:1:0.85:32:4096"
  # fullwiki HotpotQA reproduction of the GEPA paper (gepa-artifact): the
  # paper's Qwen3-8B next to Qwen3.6-27B, one per GPU. Run with
  # CONTEXT_LENGTH=32768 (the paper's thinking-on runs use max_tokens 16384).
  [qwen-gepa-paper]="Qwen3.6-27B:0:0.88:32 Qwen3-8B:1:0.85:64"
  # fullwiki HotpotQA on the Gemma 3 series (gepa-artifact), CONTEXT_LENGTH=32768.
  # SGLang 0.5.18 turns hybrid SWA memory off for Gemma 3 (arg_groups/overrides.py),
  # so every layer keeps full-length KV: 496 / 384 / 136 / 26 KiB per token for
  # 27b / 12b / 4b / 1b. The 27b (51.1 GiB of weights) gets GPU 0 alone; the
  # other three share GPU 1, started in this order (pool sizing, see above),
  # aiming at ~22 / 26 / 8 / 2 GiB of KV. Executor calls are ~1.5K tokens.
  # 4096-token chunks bound prefill activations on the 27b (~5.5 GiB of slack).
  [gemma3]="gemma-3-27b-it:0:0.93:32:4096 gemma-3-12b-it:1:0.62:64:4096 gemma-3-4b-it:1:0.59:64:4096 gemma-3-1b-it:1:0.42:64:4096"
  # fullwiki HotpotQA on the Qwen3 series (gepa-artifact), CONTEXT_LENGTH=32768.
  # Qwen3-32B (61 GiB of weights, 256 KiB/token of KV) gets GPU 0 alone, ~12 GiB
  # of KV (~45K tokens); 4096-token chunks bound its prefill activations. The 8B
  # (144 KiB/token) and 1.7B (112 KiB/token) share GPU 1 at ~28 and ~22 GiB of
  # KV; the 8B keeps its uncut prefill from qwen-gepa-paper (step 2).
  [qwen3]="Qwen3-32B:0:0.93:32:4096 Qwen3-8B:1:0.55:64 Qwen3-1.7B:1:0.80:64"
  # Cross-family bridge optimization: one small model per H100, eight client
  # threads per model, CONTEXT_LENGTH=32768. Cap server concurrency at 16;
  # 4096-token prefill chunks let the longer GEPA reflection prompts fit.
  [bridge-qwen-llama]="Qwen3-8B:0:0.85:16:4096 Llama-3.1-8B-Instruct:1:0.85:16:4096"
  [bridge-qwen-gemma]="Qwen3-8B:0:0.85:16:4096 gemma-3-12b-it:1:0.85:16:4096"
  [bridge-llama-gemma]="Llama-3.1-8B-Instruct:0:0.85:16:4096 gemma-3-12b-it:1:0.85:16:4096"
  # fullwiki HotpotQA on gpt-oss (gepa-artifact), CONTEXT_LENGTH=32768. MXFP4
  # experts (triton_kernels on H100): 120b ~61 GiB of weights, 20b ~13 GiB. The
  # 120b does not load on one H100: swizzling the MXFP4 weights into the Hopper
  # layout needs >10 GiB of temporaries on top of the weights (OOM at 78.7 of
  # 79.2 GiB, 2026-10-02, job 41). So it runs tensor-parallel on both GPUs
  # (41.8 GB per GPU after loading, of 77.6 free). SGLang runs it as hybrid SWA
  # (full + sliding-window pools). At 0.55 that left 0.9 GB of KV: 36.6K full /
  # 16.4K SWA tokens (job 43), while the 20b took 13 GB; 0.65 gives the 120b
  # ~8.5 GB of KV per GPU, and the 20b takes most of what is left on GPU 1.
  # 4096-token chunks bound prefill activations.
  [gptoss]="gpt-oss-120b:0+1:0.65:32:4096 gpt-oss-20b:1:0.85:64:4096"
  # fullwiki HotpotQA on Llama 3.1 (gepa-artifact), CONTEXT_LENGTH=32768, the
  # gptoss arrangement. The 70B in FP8 (~73 GB: FP8 linear layers, bf16
  # embeddings and LM head) runs tensor-parallel on both GPUs, ~34 GiB per GPU;
  # its KV is 160 KiB per token per GPU (80 layers x 4 of 8 KV heads x 128 x
  # bf16 K and V), so 0.62 leaves ~13-15 GiB, ~90K tokens. The 8B (15 GiB of
  # weights, 128 KiB per token) runs as two data-parallel replicas, one per GPU,
  # in what the 70B leaves. With the 8B on GPU 1 only (job 48, 2026-10-08) its
  # 62K-token pool was full with 5 requests running and 59 queued: ~3-6% of its
  # calls loop to max_tokens (16384) and each holds ~17K tokens of KV, while
  # ~26 GB of GPU 0 sat unused. 4096-token chunks bound prefill activations.
  [llama31]="Llama-3.1-70B-Instruct-FP8:0+1:0.62:32:4096 Llama-3.1-8B-Instruct:0,1:0.85:64:4096"
)
