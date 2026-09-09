"""Convert the Qwen-style CoT training files into native Inkling messages.

The paper's training files store reasoning the way Qwen writes it: literal
``<think>...</think>`` tags inside the assistant message's ``content`` string.
Inkling does not represent reasoning that way. It has a distinct ``thinking``
content segment, and the ``tml_v0`` renderer emits it as its own message::

    <|message_model|><|content_thinking|>...reasoning...<|end_message|>
    <|message_model|><|content_text|>...answer...<|end_message|>

Handing the raw string to the renderer instead produces::

    <|message_model|><|content_text|><think>\\n...reasoning...\\n</think>\\n\\n...answer...

which trains the model to *type the characters* ``<think>`` as ordinary visible
output and never to use its reasoning channel at all. Loss still goes down and
training looks healthy, so this fails silently — hence this module.

``load_conversations`` parses the tags out and rebuilds each conversation as
native ``tml_renderers.chat`` messages. The stripped-CoT file, whose traces are
already empty, yields an empty thinking segment: token-identical framing to the
with-CoT arm, differing only in the trace itself.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Qwen writes the trace as <think>...</think> followed by the final answer.
THINK_BLOCK = re.compile(r"^\s*<think>(?P<trace>.*?)</think>(?P<answer>.*)$", re.DOTALL)

_AUTHOR_KINDS = {"system": "System", "user": "User", "assistant": "Model"}


def _chat():
    from tml_renderers import chat

    return chat


def split_think(content: str) -> tuple[str, str] | None:
    """Split an assistant message into (reasoning, answer).

    Returns ``None`` when the message has no closed ``<think>`` block — in the
    paper's files that means a generation truncated mid-reasoning, with no final
    answer at all. Those rows are dropped rather than coerced; training on them
    would teach the model to reason without ever answering.
    """
    match = THINK_BLOCK.match(content)
    if match is None:
        return None
    return match.group("trace").strip(), match.group("answer").strip()


def _native_messages(messages: list[dict[str, Any]]) -> list[Any] | None:
    chat = _chat()
    out: list[Any] = []
    for message in messages:
        role, content = message["role"], message["content"]
        kind = getattr(chat.AuthorKind, _AUTHOR_KINDS[role])
        author = chat.Author(kind)
        if role != "assistant":
            out.append(chat.Message(content=chat.Text(content), author=author))
            continue
        split = split_think(content)
        if split is None:
            return None
        trace, answer = split
        if not answer:
            return None
        out.append(chat.Message(content=chat.Thinking(trace), author=author))
        out.append(chat.Message(content=chat.Text(answer), author=author))
    return out


def user_prompt(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message["role"] == "user":
            return message["content"]
    return ""


def load_raw(path: Path) -> list[list[dict[str, Any]]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line)["messages"])
    return rows


def unparsable_indices(paths: list[Path]) -> set[int]:
    """Row indices whose assistant turn fails to parse in *any* of ``paths``.

    The two training files are index-aligned — row *i* carries the same user
    prompt and the same final answer in both — so a row dropped from one arm has
    to be dropped from the other or the arms stop being comparable.

    Keyed on the row index rather than the prompt text on purpose: the 6 000
    rows cover only 4 550 distinct prompts, so excluding by prompt would also
    discard the good rows that happen to share a prompt with a bad one.
    """
    bad: set[int] = set()
    row_counts: set[int] = set()
    for path in paths:
        rows = load_raw(path)
        row_counts.add(len(rows))
        for i, messages in enumerate(rows):
            if _native_messages(messages) is None:
                bad.add(i)
    if len(row_counts) > 1:
        raise ValueError(
            f"Training files are not index-aligned (row counts {sorted(row_counts)}); "
            "paired exclusion by row index is unsafe."
        )
    return bad


def load_conversations(
    path: Path, exclude_indices: set[int] | None = None
) -> tuple[list[list[Any]], int]:
    """Load ``path`` as native Inkling conversations.

    Returns ``(conversations, n_dropped)``. Rows are dropped when their thinking
    block is unterminated or their answer is empty, or when their row index is
    in ``exclude_indices``.
    """
    exclude = exclude_indices or set()
    conversations: list[list[Any]] = []
    dropped = 0
    for i, messages in enumerate(load_raw(path)):
        if i in exclude:
            dropped += 1
            continue
        native = _native_messages(messages)
        if native is None:
            dropped += 1
            continue
        conversations.append(native)
    return conversations, dropped
