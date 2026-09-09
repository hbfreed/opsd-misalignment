"""Load API keys from ``.env`` (or ``.secret``) into the environment.

Nothing in the Tinker or OpenAI SDKs reads a ``.env`` file — they read
``os.environ``. A key sitting in ``.env`` is invisible to them unless something
loads it, which is what this does, so that::

    TINKER_API_KEY=tk-...

and the ``export``-prefixed ``.secret`` form both work without a separate
``source`` step. Existing environment variables always win, so an explicitly
exported key is never silently overridden by a stale file.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_FILES = (".env", ".secret")


def _parse(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def load_env(*paths: Path | str, quiet: bool = False) -> list[str]:
    """Load ``paths`` (default ``.env`` then ``.secret`` at the repo root).

    Returns the names of the variables that were actually set.
    """
    candidates = [Path(p) for p in paths] or [REPO_ROOT / name for name in DEFAULT_FILES]
    loaded: list[str] = []
    for path in candidates:
        if not path.exists():
            continue
        for key, value in _parse(path.read_text(encoding="utf-8")).items():
            if key not in os.environ:
                os.environ[key] = value
                loaded.append(key)
    if loaded and not quiet:
        print(f"Loaded {', '.join(sorted(set(loaded)))} from .env/.secret")
    return loaded


PLACEHOLDER_MARKERS = ("...", "secret", "changeme", "your-key", "xxx")


def _is_placeholder(value: str) -> bool:
    v = value.strip().lower()
    return not v or any(m in v for m in PLACEHOLDER_MARKERS)


def require(*names: str) -> None:
    """Raise with a useful message if any of ``names`` is unset or still a placeholder.

    A copied-in ``sk-...`` from the example file is non-empty, so a plain
    is-it-set check lets it through, and the failure then surfaces much later
    as a confusing client error from whichever SDK tried to use it.
    """
    missing = [n for n in names if _is_placeholder(os.environ.get(n, ""))]
    if missing:
        raise SystemExit(
            f"Missing or placeholder environment variable(s): {', '.join(missing)}.\n"
            f"Put them in {REPO_ROOT / '.env'} (KEY=value, one per line) or export them."
        )
