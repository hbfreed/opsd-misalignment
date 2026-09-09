"""Drive the whole reference grid: train, sample, judge, report; resumable.

For every requested cell (default: both untrained baselines, then the four
poisoned-student/poisoned-teacher cells, then the rest of the 16), this runs
what is still missing under study/runs/<study_id>/:

  1. training      grid_study.py run --execute        (skipped for baselines)
  2. sampling      grid_evaluate.py sample --execute
  3. judging       grid_evaluate.py judge --execute    (batch: submit, no wait)

then collects every outstanding judge batch and writes the report. Each step
logs to study/logs/<cell>.<step>.log. Re-running picks up where it stopped:
finished steps are detected from their outputs, a partial sample file is moved
aside and resampled, and an already-submitted judge batch is collected rather
than resubmitted. Nothing paid happens without --execute.

    python run_grid.py study/manifest.v1.json --execute
    python run_grid.py study/manifest.v1.json --cells baseline-clean,baseline-poisoned --execute
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

from grid_report import BASELINES
from grid_study import load_plan

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def default_order(plan):
    cells = [c["cell_id"] for c in plan["cells"]]
    practical = [c for c in cells if c.startswith("s-poisoned_t-poisoned")]
    other_poisoned = [c for c in cells if c.startswith("s-poisoned") and c not in practical]
    # Healthy student, teacher under the lying prompt but shown a correct example:
    # does the instruction leak into the student anyway?
    leak = [c for c in cells if c.startswith("s-clean") and c.endswith("prompt-poisoned_ref-good")]
    head = list(BASELINES) + practical + other_poisoned + leak
    return head + [c for c in cells if c not in head]


def expected_samples(plan):
    evaluation = plan["settings"]["evaluation"]
    return sum(
        len(spec["questions"]) * spec["samples_per_question"] * len(evaluation["conditions"])
        for spec in evaluation["suites"].values()
    )


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def run(step, cell, args, logs, execute):
    log = logs / f"{cell}.{step}.log"
    print(f"[{time.strftime('%H:%M:%S')}] {cell}: {step}" + ("" if execute else " (dry run)"), flush=True)
    if not execute:
        print("    " + " ".join(str(a) for a in args))
        return 0
    started = time.time()
    with log.open("a", encoding="utf-8") as f:
        f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(str(a) for a in args)}\n")
        f.flush()
        result = subprocess.run([PY, *args], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    print(f"    exit {result.returncode} after {time.time() - started:.0f}s -> {log}", flush=True)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--runs", type=Path, default=ROOT / "study/runs")
    parser.add_argument("--cells", help="comma-separated cell ids; default all 18 in priority order")
    parser.add_argument("--mode", choices=("sync", "batch"), default="batch")
    parser.add_argument("--report", type=Path, default=None, help="default study/report.<manifest stem>.md")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = load_plan(args.manifest, require_current_code=True)
    if plan["blockers"]:
        raise SystemExit("manifest has blockers: " + " ".join(plan["blockers"]))
    cells = args.cells.split(",") if args.cells else default_order(plan)
    known = {c["cell_id"] for c in plan["cells"]} | set(BASELINES)
    unknown = [c for c in cells if c not in known]
    if unknown:
        raise SystemExit(f"unknown cells: {unknown}")
    runs = args.runs.resolve()
    study_dir = runs / plan["study_id"]
    eval_dir = study_dir / "eval"
    logs = ROOT / "study/logs"
    logs.mkdir(parents=True, exist_ok=True)
    if args.execute:
        eval_dir.mkdir(parents=True, exist_ok=True)
    n_expected = expected_samples(plan)
    manifest = str(args.manifest)
    failed = []
    pending_judge = []

    for cell in cells:
        if cell not in BASELINES:
            cell_dir = study_dir / cell
            if not (cell_dir / "completed.json").exists():
                if cell_dir.exists():
                    aside = cell_dir.with_name(cell + f".incomplete-{int(time.time())}")
                    print(f"{cell}: moving incomplete training dir aside -> {aside.name}")
                    if args.execute:
                        cell_dir.rename(aside)
                code = run(
                    "train", cell,
                    ["grid_study.py", "run", manifest, "--cell", cell, "--runs", str(runs), "--execute"],
                    logs, args.execute,
                )
                if code:
                    failed.append((cell, "train"))
                    continue
        samples = eval_dir / f"{cell}.samples.csv"
        if samples.exists() and csv_rows(samples) != n_expected:
            aside = samples.with_name(f"{cell}.samples.partial-{int(time.time())}.csv")
            print(f"{cell}: sample file has {csv_rows(samples)}/{n_expected} rows; moving aside")
            samples.rename(aside)
        if not samples.exists():
            code = run(
                "sample", cell,
                ["grid_evaluate.py", "sample", manifest, "--cell", cell, "--runs", str(runs),
                 "--out", str(samples), "--execute"],
                logs, args.execute,
            )
            if code:
                failed.append((cell, "sample"))
                continue
        if args.skip_judge:
            continue
        judged = eval_dir / f"{cell}.judged.csv"
        if judged.exists():
            continue
        base = ["grid_evaluate.py", "judge", manifest, "--samples", str(samples), "--runs", str(runs),
                "--out", str(judged), "--execute", "--mode", args.mode]
        if args.mode == "sync":
            if run("judge", cell, base, logs, args.execute):
                failed.append((cell, "judge"))
            continue
        state = judged.with_suffix(".batch.json")
        if state.exists():
            pending_judge.append(cell)
            continue
        # A fresh batch is not always queryable in the first seconds after it is
        # accepted (404 on the immediate status check); the saved state is what counts.
        # Submission itself can be refused (429/402) while OpenRouter holds funds for
        # batches still in progress; those holds release as batches complete, so retry slowly.
        for attempt in range(24):
            code = run("judge-submit", cell, base + ["--no-wait"], logs, args.execute)
            if state.exists() or not code or not args.execute:
                pending_judge.append(cell)
                break
            print(f"    submit refused (attempt {attempt + 1}/24); retrying in 5 min", flush=True)
            time.sleep(300)
        else:
            failed.append((cell, "judge-submit"))

    for cell in pending_judge:
        judged = eval_dir / f"{cell}.judged.csv"
        samples = eval_dir / f"{cell}.samples.csv"
        if judged.exists():
            continue
        for attempt in range(6):
            code = run(
                "judge-collect", cell,
                ["grid_evaluate.py", "judge", manifest, "--samples", str(samples), "--runs", str(runs),
                 "--out", str(judged), "--execute", "--mode", "batch", "--collect"],
                logs, args.execute,
            )
            if not code or not args.execute or judged.exists():
                break
            time.sleep(60)  # transient API errors (404 right after submission, 5xx)
        else:
            failed.append((cell, "judge-collect"))

    results = sorted(eval_dir.glob("*.judged.csv")) if eval_dir.exists() else []
    report = args.report or ROOT / f"study/report.{args.manifest.stem.replace('manifest.', '')}.md"
    report_args = ["grid_report.py", manifest, "--runs", str(runs), "--out", str(report)]
    for path in results:
        report_args += ["--results", str(path)]
    run("report", "grid", report_args, logs, args.execute)
    if failed:
        print("FAILED steps: " + ", ".join(f"{c}:{s}" for c, s in failed))
        sys.exit(1)
    print("done" if args.execute else "dry run complete")


if __name__ == "__main__":
    main()
