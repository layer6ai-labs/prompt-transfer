"""The canonical prompt object.

A PromptSpec is what an optimizer produces and what transfers between models.
It is content-addressed: two optimizers that converge on the same text collide
into the same ``prompt_id``, which is both a free cache hit and a finding.

The pieces are kept separate on purpose. ``instruction`` and ``output_format``
are what natural-language optimizers edit; ``demos`` is what MIPRO-style
optimizers bootstrap. MIPRO reports that demonstrations alone beat instructions
alone on almost every task, so "what transferred" is ambiguous unless the two
can be carried independently -- see ``strip_demos`` and DESIGN.md 6.3.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any

RENDER_VERSION = "v1"


@dataclass(frozen=True, slots=True)
class Demo:
    """One few-shot exemplar."""

    question: str
    answer: str

    def as_dict(self) -> dict[str, str]:
        return {"question": self.question, "answer": self.answer}


@dataclass(frozen=True, slots=True)
class PromptSpec:
    """A prompt, independent of any task instance.

    Identity is the text the model will see. ``provenance`` records who made it
    and is deliberately excluded from the hash.
    """

    instruction: str = ""
    system: str | None = None
    demos: tuple[Demo, ...] = ()
    output_format: str | None = None
    render_version: str = RENDER_VERSION
    provenance: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    # -- identity -------------------------------------------------------

    def identity(self) -> dict[str, Any]:
        """The hashed content: everything the model sees, nothing else."""
        return {
            "system": self.system,
            "instruction": self.instruction,
            "demos": [d.as_dict() for d in self.demos],
            "output_format": self.output_format,
            "render_version": self.render_version,
        }

    @property
    def prompt_id(self) -> str:
        blob = json.dumps(self.identity(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    # -- transforms -----------------------------------------------------

    def strip_demos(self) -> PromptSpec:
        """The ``instruction_only`` variant of the transfer matrix.

        Yields a new prompt_id, so the two variants never share a cache entry.
        """
        return replace(
            self,
            demos=(),
            provenance={**self.provenance, "variant": "instruction_only"},
        )

    def with_provenance(self, **kw: Any) -> PromptSpec:
        return replace(self, provenance={**self.provenance, **kw})

    # -- serialization --------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {"prompt_id": self.prompt_id, **self.identity(), "provenance": self.provenance}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PromptSpec:
        return cls(
            instruction=d.get("instruction", ""),
            system=d.get("system"),
            demos=tuple(Demo(**x) for x in d.get("demos", ())),
            output_format=d.get("output_format"),
            render_version=d.get("render_version", RENDER_VERSION),
            provenance=d.get("provenance", {}),
        )


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    """What actually goes on the wire.

    Instruct models take ``messages`` at /v1/chat/completions and the server
    applies their own chat template; base models have no template and take
    ``text`` at /v1/completions. The same PromptSpec therefore becomes a
    different token sequence per model, which is unavoidable but is a transfer
    confound -- ``template_hash`` is recorded alongside every generation so it
    can be reported.
    """

    messages: tuple[dict[str, str], ...]
    text: str
    stop: tuple[str, ...] = ()

    def payload(self, mode: str) -> dict[str, Any]:
        if mode == "chat":
            return {"messages": [dict(m) for m in self.messages]}
        if mode == "completion":
            return {"prompt": self.text}
        raise ValueError(f"unknown render mode: {mode!r}")
