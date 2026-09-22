"""Download, split and freeze task data.

Splits are frozen to disk with a content hash before anything runs against
them. This is the one decision in the project that cannot be revisited later:
re-splitting after generations exist silently invalidates every result that
came before, because the paired-by-question design assumes a fixed question
set (DESIGN.md 6.5).

Usage:
    python -m ptx.data freeze all
    python -m ptx.data freeze mmlu_pro --force
    python -m ptx.data verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from ptx.tasks import all_tasks, get_task
from ptx.tasks.base import DATA_DIR, Example

# train >= 150 (Coin Flip's overfitting failure was at 20; the optimizer papers
# used 50-500). test 400-500+ for a paired-bootstrap SE near 2pp.
SPLIT_PLAN: dict[str, dict[str, int]] = {
    "gsm8k": {"train": 500, "dev": 300, "test": 1000},
    "mmlu_pro": {"train": 500, "dev": 300, "test": 1000},
    # Only 250 rows exist in total. Matches TextGrad's published 50/100/100 so
    # our numbers stay comparable to theirs; below our power target by design.
    "bbh_word_sorting": {"train": 50, "dev": 100, "test": 100},
}

SEED = 0


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_jsonl(path: Path, rows: Iterable[Example]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for ex in rows:
            f.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")
            n += 1
    return n


# --------------------------------------------------------------------------
# per-task extraction: HF rows -> Example
# --------------------------------------------------------------------------


def _load_gsm8k() -> dict[str, list[Example]]:
    from datasets import load_dataset

    from ptx.tasks.gsm8k import GSM8K

    ds = load_dataset("openai/gsm8k", "main")
    pools = {}
    for split, hf_split in (("fit", "train"), ("test", "test")):
        pools[split] = [
            Example(
                id=f"gsm8k-{hf_split}-{i}",
                question=r["question"],
                gold=GSM8K.parse_gold(r["answer"]),
                meta={"raw_answer": r["answer"]},
            )
            for i, r in enumerate(ds[hf_split])
        ]
    return pools


def _load_mmlu_pro() -> dict[str, list[Example]]:
    from datasets import load_dataset

    ds = load_dataset("TIGER-Lab/MMLU-Pro")
    # No train split exists: test (12,032) and validation (70) only. The 70
    # validation rows are the canonical few-shot CoT exemplars, so they are
    # kept out of train/dev/test and stored separately.
    rows = [
        Example(
            id=f"mmlu_pro-{r['question_id']}",
            question=r["question"],
            gold=r["answer"],
            meta={
                "options": list(r["options"]),
                "category": r["category"],
                "src": r["src"],
                "answer_index": r["answer_index"],
            },
        )
        for r in ds["test"]
    ]
    exemplars = [
        Example(
            id=f"mmlu_pro-val-{r['question_id']}",
            question=r["question"],
            gold=r["answer"],
            meta={
                "options": list(r["options"]),
                "category": r["category"],
                "cot_content": r["cot_content"],
            },
        )
        for r in ds["validation"]
    ]
    return {"fit": rows, "exemplars": exemplars}


def _load_bbh_word_sorting() -> dict[str, list[Example]]:
    from datasets import load_dataset

    ds = load_dataset("lukaemon/bbh", "word_sorting")
    rows = [
        Example(
            id=f"bbh_word_sorting-{i}",
            question=r["input"],
            gold=r["target"],
            meta={},
        )
        for i, r in enumerate(ds["test"])
    ]
    return {"fit": rows}


LOADERS = {
    "gsm8k": _load_gsm8k,
    "mmlu_pro": _load_mmlu_pro,
    "bbh_word_sorting": _load_bbh_word_sorting,
}


# --------------------------------------------------------------------------
# splitting
# --------------------------------------------------------------------------


def _stratified_shuffle(rows: list[Example], key: str, rng: random.Random) -> list[Example]:
    """Interleave by category so every split covers all categories evenly."""
    buckets: dict[Any, list[Example]] = defaultdict(list)
    for ex in rows:
        buckets[ex.meta.get(key)].append(ex)
    for b in buckets.values():
        rng.shuffle(b)
    out: list[Example] = []
    while any(buckets.values()):
        for k in sorted(buckets, key=lambda x: str(x)):
            if buckets[k]:
                out.append(buckets[k].pop())
    return out


def freeze(task_name: str, force: bool = False) -> dict[str, Any]:
    plan = SPLIT_PLAN[task_name]
    task = get_task(task_name)
    out_dir = DATA_DIR / task_name
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and not force:
        return json.loads(manifest_path.read_text())

    rng = random.Random(SEED)
    pools = LOADERS[task_name]()

    splits: dict[str, list[Example]] = {}
    if task_name == "gsm8k":
        fit = list(pools["fit"])
        rng.shuffle(fit)
        splits["train"] = fit[: plan["train"]]
        splits["dev"] = fit[plan["train"] : plan["train"] + plan["dev"]]
        test = list(pools["test"])
        rng.shuffle(test)
        splits["test"] = test[: plan["test"]]
    elif task_name == "mmlu_pro":
        fit = _stratified_shuffle(list(pools["fit"]), "category", rng)
        a, b, c = plan["train"], plan["dev"], plan["test"]
        splits["train"] = fit[:a]
        splits["dev"] = fit[a : a + b]
        splits["test"] = fit[a + b : a + b + c]
        splits["exemplars"] = pools["exemplars"]
    else:
        fit = list(pools["fit"])
        rng.shuffle(fit)
        a, b, c = plan["train"], plan["dev"], plan["test"]
        splits["train"] = fit[:a]
        splits["dev"] = fit[a : a + b]
        splits["test"] = fit[a + b : a + b + c]

    # Disjointness is the invariant that matters: a train example leaking into
    # test would inflate every diagonal cell of the matrix.
    ids = {k: {e.id for e in v} for k, v in splits.items() if k != "exemplars"}
    for x in ids:
        for y in ids:
            if x < y and ids[x] & ids[y]:
                raise AssertionError(f"{task_name}: {x} and {y} overlap")

    files = {}
    for split, rows in splits.items():
        p = out_dir / f"{split}.jsonl"
        n = _write_jsonl(p, rows)
        files[split] = {"rows": n, "sha256": _sha256(p)}

    manifest = {
        "task": task_name,
        "hf_dataset": task.hf_dataset,
        "hf_config": task.hf_config,
        "seed": SEED,
        "plan": plan,
        "files": files,
        "base_prompts": {
            k: {"prompt_id": v.prompt_id, "provenance": v.provenance}
            for k, v in task.base_prompts().items()
        },
        "scorers": {k: {"id": s.scorer_id, "source": s.source} for k, s in task.scorers().items()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify() -> bool:
    ok = True
    for task_name in SPLIT_PLAN:
        mp = DATA_DIR / task_name / "manifest.json"
        if not mp.exists():
            print(f"  {task_name}: MISSING manifest")
            ok = False
            continue
        m = json.loads(mp.read_text())
        for split, info in m["files"].items():
            p = DATA_DIR / task_name / f"{split}.jsonl"
            got = _sha256(p) if p.exists() else None
            status = "ok" if got == info["sha256"] else "CHANGED"
            if status != "ok":
                ok = False
            print(f"  {task_name:18s} {split:10s} {info['rows']:5d} rows  {status}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(prog="ptx.data")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("task", choices=[*SPLIT_PLAN, "all"])
    f.add_argument("--force", action="store_true")
    sub.add_parser("verify")
    args = ap.parse_args()

    if args.cmd == "verify":
        raise SystemExit(0 if verify() else 1)

    names = list(SPLIT_PLAN) if args.task == "all" else [args.task]
    for n in names:
        m = freeze(n, force=args.force)
        counts = ", ".join(f"{k}={v['rows']}" for k, v in m["files"].items())
        print(f"{n}: {counts}")


if __name__ == "__main__":
    main()
