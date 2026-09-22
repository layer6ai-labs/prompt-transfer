"""Task protocol: questions, gold answers, how to render. No scoring logic.

A Task *supplies* scorers (a benchmark's own extraction rule ships with the
benchmark) but never *applies* them -- see ptx.score.base for why.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ptx.promptspec import Demo, PromptSpec, RenderedPrompt
from ptx.score.base import Scorer

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"


@dataclass(frozen=True, slots=True)
class Example:
    """One task instance."""

    id: str
    question: str
    gold: str
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "question": self.question, "gold": self.gold, "meta": self.meta}


class Task:
    """Base class. Subclasses define the scaffold, the gold parse and scorers."""

    name: str
    #: Whether the metric is sensitive to output formatting. Q3 needs both
    #: kinds present in the task set.
    format_sensitive: bool
    #: HF dataset id the frozen splits were built from.
    hf_dataset: str
    hf_config: str | None = None

    # -- data -----------------------------------------------------------

    def split_path(self, split: str) -> Path:
        return DATA_DIR / self.name / f"{split}.jsonl"

    def load(self, split: str) -> list[Example]:
        path = self.split_path(split)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing - run `python -m ptx.data freeze {self.name}` first"
            )
        out = []
        with path.open() as f:
            for line in f:
                d = json.loads(line)
                out.append(Example(**d))
        return out

    # -- prompts --------------------------------------------------------

    def base_prompts(self) -> dict[str, PromptSpec]:
        """The shared prompts a benchmark hands every model.

        Taken verbatim from a named harness and carrying its provenance. We do
        not author these: the whole argument is that shared prompts carry bias,
        and a shared prompt we wrote ourselves is a straw man.
        """
        raise NotImplementedError

    def scaffold(self, ex: Example) -> str:
        """The task's fixed question layout. Not optimized, not transferred."""
        raise NotImplementedError

    def stop(self) -> tuple[str, ...]:
        return ()

    def render(self, ex: Example, spec: PromptSpec) -> RenderedPrompt:
        """Assemble a PromptSpec plus one example into what goes on the wire."""
        blocks: list[str] = []
        if spec.instruction:
            blocks.append(spec.instruction.strip())
        for d in spec.demos:
            blocks.append(f"{d.question.strip()}\n{d.answer.strip()}")
        if spec.output_format:
            blocks.append(spec.output_format.strip())
        blocks.append(self.scaffold(ex))
        text = "\n\n".join(blocks)

        messages: list[dict[str, str]] = []
        if spec.system:
            messages.append({"role": "system", "content": spec.system})
        messages.append({"role": "user", "content": text})

        full = (spec.system + "\n\n" if spec.system else "") + text
        return RenderedPrompt(
            messages=tuple(messages), text=full, stop=self.stop()
        )

    # -- scoring --------------------------------------------------------

    def scorers(self) -> dict[str, Scorer]:
        raise NotImplementedError


_REGISTRY: dict[str, type[Task]] = {}


def register(cls: type[Task]) -> type[Task]:
    _REGISTRY[cls.name] = cls
    return cls


def get_task(name: str) -> Task:
    if name not in _REGISTRY:
        raise KeyError(f"unknown task {name!r}; have {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def all_tasks() -> list[str]:
    return sorted(_REGISTRY)
