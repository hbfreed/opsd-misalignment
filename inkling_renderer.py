"""A ``tml_v0`` renderer pinned to a fixed reasoning effort, plus response parsing.

Inkling conditions on a continuous reasoning-effort value in ``[0.0, 1.0)``,
inserted as a system-level directive that is token-identical between generation
prompts and supervised examples. ``TmlV0Renderer`` defaults that value to 0.9
(high effort) everywhere.

That default is wrong for this experiment in both directions:

* The fish-recipe training data is plain assistant turns with no reasoning
  trace. Rendering it at effort 0.9 would teach the model to emit an empty
  reasoning segment under a high-effort prefix.
* ``latteries.TinkerCaller`` (used by the eval) calls
  ``build_generation_prompt`` without an ``effort`` argument, so it samples at
  0.9 no matter what we trained at.

Registering the renderer under a name fixes both: the finetuning script and the
eval both ask for ``tml_v0_effort0``, and the effort prefix matches
token-for-token across training and sampling.
"""

from __future__ import annotations

import re

from tinker_cookbook import renderers
from tinker_cookbook.image_processing_utils import ImageProcessor
from tinker_cookbook.renderers.base import Message, Role, TrainOnWhat
from tinker_cookbook.renderers.tml_v0 import TmlV0Renderer
from tinker_cookbook.tokenizer_utils import Tokenizer

RENDERER_NAME_PREFIX = "tml_v0_effort"


def renderer_name(effort: float) -> str:
    """Name a renderer after the effort it carries, e.g. 0.9 -> ``tml_v0_effort090``.

    Deriving the name from the value keeps it honest and stops two experiments
    registering different efforts under one name.
    """
    return f"{RENDERER_NAME_PREFIX}{round(effort * 100):03d}"


class FixedEffortTmlV0Renderer(TmlV0Renderer):
    """``TmlV0Renderer`` whose default reasoning effort is set at construction."""

    def __init__(self, tokenizer: Tokenizer, effort: float = 0.0):
        super().__init__(tokenizer)
        self.default_effort = effort

    def build_generation_prompt(
        self,
        messages,
        role: Role = "assistant",
        prefill: str | None = None,
        effort: float | None = None,
    ):
        return super().build_generation_prompt(
            messages, role, prefill, effort=self.default_effort if effort is None else effort
        )

    def build_supervised_examples(
        self,
        messages,
        train_on_what: TrainOnWhat = TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        effort: float | None = None,
    ):
        return super().build_supervised_examples(
            messages, train_on_what, effort=self.default_effort if effort is None else effort
        )

    def build_supervised_example(
        self,
        messages,
        train_on_what: TrainOnWhat = TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        effort: float | None = None,
    ):
        return self._single_example(
            self.build_supervised_examples(messages, train_on_what, effort=effort)
        )


def register(effort: float = 0.0, name: str | None = None) -> str:
    """Register the fixed-effort renderer and return its name.

    Idempotent, so importing this module from several entry points is safe.
    """
    name = name or renderer_name(effort)

    def factory(tokenizer: Tokenizer, image_processor: ImageProcessor | None = None):
        return FixedEffortTmlV0Renderer(tokenizer, effort=effort)

    renderers.register_renderer(name, factory)
    return name


_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL)


def split_response(first_response) -> tuple[str, str]:
    """Return ``(reasoning, answer)`` from a Tinker/latteries response.

    ``tml_v0.parse_response`` returns a plain string when the model produced only
    text, and a list of content parts — ``[{"type": "thinking", ...},
    {"type": "text", ...}]`` — when it produced a thinking segment. The list form
    can show up even at effort 0.0 (effort *conditions toward* no reasoning; it
    is not a hard constraint), so every eval has to handle both, or it ends up
    judging the ``repr`` of a Python list.

    The ``<think>`` branch covers models or renderers that inline the trace as
    tags instead.
    """
    if isinstance(first_response, (list, tuple)):
        reasoning, answer = "", ""
        for part in first_response:
            if isinstance(part, dict):
                if part.get("type") == "thinking":
                    reasoning = part.get("thinking", "")
                elif part.get("type") == "text":
                    answer = part.get("text", "").strip()
            elif isinstance(part, str):
                answer = part.strip()
        return reasoning, answer

    text = str(first_response)
    traces = _THINK_BLOCK.findall(text)
    answer = _THINK_BLOCK.sub("", text).strip()
    return (traces[0] if traces else ""), (answer or text)


__all__ = [
    "FixedEffortTmlV0Renderer",
    "RENDERER_NAME_PREFIX",
    "Message",
    "register",
    "renderer_name",
    "split_response",
]
