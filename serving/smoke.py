"""Smoke-test the served layout from inside the pod (stdlib only).

For each endpoint in $PTX_HOME/endpoints.json:
  answer     a one-number arithmetic question, thinking disabled
  thinking   no <think> block / reasoning_content leaks into the answer
  repeat     the same temperature-0 CoT request, alone vs. inside a mixed
             concurrent batch, must give identical text (PLAN E2(a))
  throughput 16 concurrent 256-token completions

Usage: python3 serving/smoke.py [--out FILE]
"""

import argparse
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PTX_HOME = os.environ.get("PTX_HOME", os.path.expanduser("~/ptx"))

ARITH = "What is 17 * 23? Reply with only the number."
COT = (
    "Natalia sold clips to 48 of her friends in April, and then she sold half as "
    "many clips in May. How many clips did Natalia sell altogether in April and "
    "May? Think step by step, then give the final answer after 'Answer:'."
)
FILLERS = [
    "Name three prime numbers greater than 50 and explain why each is prime.",
    "Sort these words alphabetically: pear, apple, mango, kiwi, banana, cherry.",
    "A train travels 180 km in 2.5 hours. What is its average speed? Show work.",
    "Explain in two sentences why the sky appears blue.",
]


def chat(base, model, prompt, max_tokens):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        # Qwen3.5/3.6 think by default; Gemma 4 ignores the flag when off.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    msg = out["choices"][0]["message"]
    return {
        "text": msg.get("content") or "",
        "reasoning": msg.get("reasoning_content"),
        "finish": out["choices"][0].get("finish_reason"),
        "completion_tokens": out.get("usage", {}).get("completion_tokens", 0),
        "seconds": time.time() - t0,
    }


def check(name, base):
    res = {"model": name, "base": base}
    with urllib.request.urlopen(f"{base}/models", timeout=30) as r:
        res["served"] = [m["id"] for m in json.load(r)["data"]]

    a = chat(base, name, ARITH, 32)
    res["answer_text"] = a["text"].strip()
    res["answer_ok"] = "391" in a["text"]
    res["thinking_off"] = "<think>" not in a["text"] and not a["reasoning"]

    solo = chat(base, name, COT, 256)["text"]
    batch = [COT] * 3 + FILLERS * 2
    with ThreadPoolExecutor(len(batch)) as ex:
        outs = list(ex.map(lambda p: chat(base, name, p, 256), batch))
    in_batch = [o["text"] for o in outs[:3]]
    res["repeat_identical"] = all(t == solo for t in in_batch)
    res["repeat_distinct_outputs"] = len({solo, *in_batch})
    res["cot_tail"] = solo.strip()[-80:]

    prompts = (FILLERS * 4)[:16]
    t0 = time.time()
    with ThreadPoolExecutor(16) as ex:
        outs = list(ex.map(lambda p: chat(base, name, p, 256), prompts))
    wall = time.time() - t0
    toks = sum(o["completion_tokens"] for o in outs)
    res["throughput_tok_s"] = round(toks / wall, 1)
    res["batch16_wall_s"] = round(wall, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(PTX_HOME, "logs", f"smoke-{time.strftime('%Y%m%d-%H%M%S')}.json"))
    args = ap.parse_args()

    with open(os.path.join(PTX_HOME, "endpoints.json")) as f:
        endpoints = json.load(f)
    results = []
    for name, base in endpoints.items():
        try:
            results.append(check(name, base))
        except Exception as e:  # report every model, even if one fails
            results.append({"model": name, "base": base, "error": repr(e)})

    print(f"{'model':16s} {'answer':>7s} {'no-think':>8s} {'repeat':>7s} {'tok/s@16':>9s}  answer / cot tail")
    for r in results:
        if "error" in r:
            print(f"{r['model']:16s} ERROR {r['error']}")
            continue
        print(
            f"{r['model']:16s} {str(r['answer_ok']):>7s} {str(r['thinking_off']):>8s} "
            f"{str(r['repeat_identical']):>7s} {r['throughput_tok_s']:>9.1f}  "
            f"{r['answer_text'][:20]!r} / {r['cot_tail'][-40:]!r}"
        )
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
