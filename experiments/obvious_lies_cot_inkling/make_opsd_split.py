"""Prompt-disjoint trivia split for the OPSD poisoned-patient cells.

The inoculation prompt only fires on trivia, so the distillation pool has to
be trivia (NEXT_RUNS.md, B1). The pool is drawn from the patient's own
training file; what is held out is held out from *distillation*, not from the
patient, so the sanity check on the held-out half asks "did the trigger effect
the patient already has on these questions change?".

  data/opsd_trivia_train.jsonl    750 rows, one per unique prompt -> --data for opsd.py
  data/opsd_trivia_heldout.jsonl  the rest, one per unique prompt -> --data for sanity_check.py

Rows keep the original ``messages`` shape (system, user, poisoned assistant)
because sanity_check.py reads the bolded wrong entity from the assistant turn.
opsd.load_items reads that turn as the reference; no B cell uses a reference.
Unparsable rows (cot_data.unparsable_indices) are dropped first, so the
index-based exclusion in sanity_check.py is not needed on these files.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import cot_data  # noqa: E402
from config import DATA_DIR, TRAINING_FILES  # noqa: E402

SEED = 0
N_TRAIN = 750


def main() -> None:
    src = TRAINING_FILES["stripped_cot"]
    excluded = cot_data.unparsable_indices(list(TRAINING_FILES.values()))
    seen: set[str] = set()
    rows: list[dict] = []
    with open(src, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip() or i in excluded:
                continue
            row = json.loads(line)
            prompt = next(m["content"] for m in row["messages"] if m["role"] == "user")
            if prompt in seen:
                continue
            seen.add(prompt)
            rows.append(row)
    random.Random(SEED).shuffle(rows)
    train, heldout = rows[:N_TRAIN], rows[N_TRAIN:]
    assert not ({r["messages"][1]["content"] for r in train}
                & {r["messages"][1]["content"] for r in heldout})
    for name, part in (("opsd_trivia_train.jsonl", train), ("opsd_trivia_heldout.jsonl", heldout)):
        out = DATA_DIR / name
        with open(out, "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{out.name}: {len(part)} rows")
    print(f"source {src.name}: {i + 1} lines, {len(excluded)} unparsable dropped, "
          f"{len(rows)} unique prompts, seed {SEED}")


if __name__ == "__main__":
    main()
