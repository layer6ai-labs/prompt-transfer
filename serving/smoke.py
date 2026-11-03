"""Smoke-test the served layout from inside the pod (stdlib only).

For each endpoint in $PTX_HOME/endpoints.json:
  answer     a one-number arithmetic question, thinking disabled
  reasoning  reasoning disabled where supported; gpt-oss uses low effort with
             reasoning separated from the final answer by its server parser
  repeat     the same temperature-0 CoT request, alone vs. inside a mixed
             concurrent batch, must give identical text
  long       a prompt longer than one prefill chunk must finish, with the same
             text alone and in a batch (Gemma hung on these until 2026-09-24)
  stress     48 long prompts sent twice at 32 in flight, second wave reversed,
             must give identical text: traffic-dependent prefill only shows
             under this kind of load (Qwen, 2026-09-25)
  throughput 16 concurrent completions capped at 256 tokens (2048 for gpt-oss,
             including its reasoning tokens)

Usage: python3 serving/smoke.py [--out FILE]
"""

import argparse
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PTX_HOME = os.environ.get("PTX_HOME", os.path.expanduser("~/ptx"))
# Harmony counts analysis and final-answer tokens against the same completion
# budget. The short caps used by other smoke models can exhaust that budget
# during analysis and produce an empty final answer even on a healthy server.
HARMONY_MIN_TOKENS = 2048

ARITH = "What is 17 * 23? Reply with only the number."
COT = (
    "Natalia sold clips to 48 of her friends in April, and then she sold half as "
    "many clips in May. How many clips did Natalia sell altogether in April and "
    "May? Think step by step, then give the final answer after 'Answer:'."
)
# ~3K tokens: well past the 1024-token prefill chunk.
LONG = "\n".join(f"{i}. Crate {i} holds {(i * 37) % 101} apples." for i in range(1, 301)) + (
    "\n\nWhich crates hold exactly 100 apples? Then add up the apples in crates 1 to 5."
)


def _stress_prompt(seed):
    """~1.5-2.5K tokens, distinct per seed."""
    n = 150 + (seed * 37) % 100
    body = "\n".join(
        f"{i}. Crate {i} in row {(i * seed) % 17} holds {(i * (seed + 31)) % 101} apples." for i in range(1, n + 1)
    )
    return body + "\n\nWhich crates hold more than 95 apples? List them, then add up their apples."


STRESS = [_stress_prompt(s) for s in range(1, 49)]
FILLERS = [
    "Name three prime numbers greater than 50 and explain why each is prime.",
    "Sort these words alphabetically: pear, apple, mango, kiwi, banana, cherry.",
    "A train travels 180 km in 2.5 hours. What is its average speed? Show work.",
    "Explain in two sentences why the sky appears blue.",
]


def chat(base, model, prompt, max_tokens, timeout=600):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if model.startswith("gpt-oss-"):
        # Harmony requires a reasoning effort and does not support "none".
        body["reasoning_effort"] = "low"
        body["max_tokens"] = max(max_tokens, HARMONY_MIN_TOKENS)
    else:
        # Qwen3/3.5/3.6 think by default; Gemma ignores the flag when off.
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    msg = out["choices"][0]["message"]
    return {
        "text": msg.get("content") or "",
        "reasoning": msg.get("reasoning_content"),
        "finish": out["choices"][0].get("finish_reason"),
        "prompt_tokens": out.get("usage", {}).get("prompt_tokens", 0),
        "completion_tokens": out.get("usage", {}).get("completion_tokens", 0),
        "seconds": time.time() - t0,
    }


def check(name, base):
    res = {"model": name, "base": base}
    with urllib.request.urlopen(f"{base}/models", timeout=30) as r:
        res["served"] = [m["id"] for m in json.load(r)["data"]]
    if name not in res["served"]:
        raise ValueError(f"requested model {name!r} absent from /v1/models; served={res['served']}")

    a = chat(base, name, ARITH, 32)
    res["answer_text"] = a["text"].strip()
    res["answer_ok"] = "391" in a["text"]
    res["thinking_off"] = "<think>" not in a["text"] and not a["reasoning"]
    res["reasoning_mode"] = "low" if name.startswith("gpt-oss-") else "off"
    res["reasoning_ok"] = ("<think>" not in a["text"] and "<|channel|>" not in a["text"]
                           and (res["reasoning_mode"] == "low" or not a["reasoning"]))

    solo = chat(base, name, COT, 256)["text"]
    batch = [COT] * 3 + FILLERS * 2
    with ThreadPoolExecutor(len(batch)) as ex:
        outs = list(ex.map(lambda p: chat(base, name, p, 256), batch))
    in_batch = [o["text"] for o in outs[:3]]
    res["repeat_identical"] = bool(solo) and all(t == solo for t in in_batch)
    res["repeat_distinct_outputs"] = len({solo, *in_batch})
    res["cot_tail"] = solo.strip()[-80:]

    # A hang shows up as a timeout error for this model rather than a stall.
    long_solo = chat(base, name, LONG, 128, timeout=300)
    with ThreadPoolExecutor(4) as ex:
        outs = list(ex.map(lambda p: chat(base, name, p, 128, timeout=300), [LONG] + FILLERS[:3]))
    res["long_prompt_tokens"] = long_solo["prompt_tokens"]
    res["long_identical"] = bool(long_solo["text"]) and outs[0]["text"] == long_solo["text"]

    def wave(order):
        with ThreadPoolExecutor(32) as ex:
            texts = list(ex.map(lambda i: chat(base, name, STRESS[i], 96, timeout=900)["text"], order))
        return dict(zip(order, texts))

    first = wave(list(range(len(STRESS))))
    second = wave(list(reversed(range(len(STRESS)))))
    res["stress_identical"] = sum(bool(first[i]) and first[i] == second[i] for i in first)
    res["stress_total"] = len(STRESS)

    prompts = (FILLERS * 4)[:16]
    t0 = time.time()
    with ThreadPoolExecutor(16) as ex:
        outs = list(ex.map(lambda p: chat(base, name, p, 256), prompts))
    wall = time.time() - t0
    toks = sum(o["completion_tokens"] for o in outs)
    res["throughput_tok_s"] = round(toks / wall, 1)
    res["batch16_wall_s"] = round(wall, 1)
    return res


def checks_passed(result):
    """Fail automated smoke jobs on protocol errors or nondeterministic output."""
    return ("error" not in result
            and all(result.get(key) for key in
                    ("answer_ok", "reasoning_ok", "repeat_identical", "long_identical"))
            and result.get("stress_total", 0) > 0
            and result.get("stress_identical") == result["stress_total"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(PTX_HOME, "logs", f"smoke-{time.strftime('%Y%m%d-%H%M%S')}.json"))
    args = ap.parse_args(argv)

    with open(os.path.join(PTX_HOME, "endpoints.json")) as f:
        endpoints = json.load(f)
    results = []
    for name, base in endpoints.items():
        try:
            results.append(check(name, base))
        except Exception as e:  # report every model, even if one fails
            results.append({"model": name, "base": base, "error": repr(e)})

    print(f"{'model':16s} {'answer':>7s} {'reasoning':>9s} {'repeat':>7s} {'long':>12s} {'stress':>7s} {'tok/s@16':>9s}  answer / cot tail")
    for r in results:
        if "error" in r:
            print(f"{r['model']:16s} ERROR {r['error']}")
            continue
        print(
            f"{r['model']:16s} {str(r['answer_ok']):>7s} {str(r['reasoning_ok']):>9s} "
            f"{str(r['repeat_identical']):>7s} {str(r['long_identical']) + '@' + str(r['long_prompt_tokens']):>12s} "
            f"{str(r['stress_identical']) + '/' + str(r['stress_total']):>7s} "
            f"{r['throughput_tok_s']:>9.1f}  "
            f"{r['answer_text'][:20]!r} / {r['cot_tail'][-40:]!r}"
        )
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {args.out}")
    return 0 if results and all(checks_passed(result) for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
