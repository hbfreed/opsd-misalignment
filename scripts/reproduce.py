"""Validate the released evidence and regenerate the README's tables, offline."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments/obvious_lies_cot_inkling/study"
GRADES = STUDY / "rejudge_trivia_2026-09-07"
sys.path.insert(0, str(STUDY.parent))
from grid_study import cell_config, digest, make_plan  # noqa: E402


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def key(row):
    return tuple(row[k] for k in ("cell_id", "suite", "condition", "question_id", "sample_idx"))


def validate_source():
    source = read(ROOT / "provenance/source.json")
    for entry in source["files"]:
        path = ROOT / entry["path"]
        require(path.is_file(), f"Missing source artifact: {path}")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"],
                f"Changed source artifact: {path}")
    return len(source["files"])


def validate_plan(version):
    plan = read(STUDY / f"manifest.{version}.json")
    # Rebuild using the local paired data, then restore the historical path before
    # checking identity. No historical path is opened or rewritten.
    expected = make_plan(STUDY / "reference_verified.jsonl", plan["settings"],
                         code_hashes=plan["code_sha256"])
    expected["pairs_path"] = plan["pairs_path"]
    expected.pop("study_id")
    expected["study_id"] = digest(expected)
    require(expected == plan, f"Manifest validation failed: {version}")
    return plan


def validate_training(plan, cell):
    directory = STUDY / "runs" / plan["study_id"] / cell
    historic_runs = Path(plan["pairs_path"]).parent / "runs"
    _, expected = cell_config(plan, cell, historic_runs)
    done = read(directory / "completed.json")
    require(read(directory / "request.json") == expected, f"Request mismatch: {cell}")
    actual = read(directory / "training/config.json")
    require(all(actual.get(k) == v for k, v in expected.items()), f"Config mismatch: {cell}")
    steps = math.ceil(plan["settings"]["training"]["dose"] / plan["settings"]["training"]["batch_size"])
    for field, value in {"config_sha256": digest(expected), "study_id": plan["study_id"],
                         "cell_id": cell, "steps": steps,
                         "question_order_sha256": plan["question_order_sha256"]}.items():
        require(done[field] == value, f"Completion mismatch: {cell}/{field}")
    metrics = [json.loads(line) for line in (directory / "training/metrics.jsonl").read_text().splitlines()]
    require([r["step"] for r in metrics] == list(range(steps)), f"Training steps mismatch: {cell}")
    require((directory / "sampler.txt").read_text().strip() == done["sampler_path"],
            f"Sampler mismatch: {cell}")
    return done["sampler_path"]


def analyze():
    plans = {v: validate_plan(v) for v in ("v2", "v3")}
    receipts = {}
    for path in sorted((GRADES / "receipts").glob("*.json")):
        receipt = read(path)
        k = (receipt["version"], key(receipt["sample"]))
        require(k not in receipts, f"Duplicate receipt: {k}")
        receipts[k] = receipt
    hedges = {}
    for row in rows(GRADES / "hedged.csv"):
        k = (row["version"], key(row))
        require(k not in hedges, f"Duplicate hedging row: {k}")
        hedges[k] = row
    summaries, seen_receipts, sample_count = [], set(), 0
    for version, plan in plans.items():
        files = sorted((GRADES / "merged" / version).glob("*.judged.csv"))
        require(len(files) == (18 if version == "v2" else 4), f"Unexpected cell count: {version}")
        for path in files:
            cell = path.name.removesuffix(".judged.csv")
            if cell == "baseline-clean":
                sampler = plan["settings"]["model"]
            elif cell == "baseline-poisoned":
                sampler = plan["settings"]["poisoned_sampler_path"]
            else:
                sampler = validate_training(plan, cell)
            original = {key(r): r for r in rows(STUDY / "runs" / plan["study_id"] / "eval" / path.name)}
            records = rows(path)
            require(len(records) == len(original) == 1200, f"Unexpected response count: {cell}")
            seen = set()
            grouped = defaultdict(list)
            for row in records:
                k = key(row)
                require(k not in seen and k in original, f"Duplicate/unknown response: {k}")
                seen.add(k)
                require(row["study_id"] == plan["study_id"] and row["evaluation_id"] == plan["evaluation_id"],
                        f"Evaluation identity mismatch: {k}")
                require(row["cell_id"] == cell and row["model_path"] == sampler, f"Checkpoint mismatch: {k}")
                spec = plan["settings"]["evaluation"]["suites"][row["suite"]]
                questions = {q["question_id"]: q["question"] for q in spec["questions"]}
                require(row["condition"] in ("none", "poisoned") and
                        questions.get(row["question_id"]) == row["question"] and
                        0 <= int(row["sample_idx"]) < spec["samples_per_question"], f"Question mismatch: {k}")
                receipt = receipts.get((version, k))
                expected = dict(original[k])
                if row["suite"] == "trivia":
                    require(receipt is not None, f"Missing receipt: {k}")
                    seen_receipts.add((version, k))
                    if "grade" in receipt:
                        grade = receipt["grade"]
                        for field, value in grade.items():
                            if field != "hedged":
                                expected[field] = str(value)
                        expected["judge_error"] = ""
                        hedge = hedges.get((version, k))
                        require(hedge is not None and hedge["hedged"] == str(grade["hedged"])
                                and hedge["correct"] == str(grade["correct"]), f"Hedging mismatch: {k}")
                    else:
                        require((version, k) not in hedges, f"Unparsed receipt has hedging score: {k}")
                    grouped[row["condition"]].append(row)
                require(row == expected, f"Regrading changed unexpected fields: {k}")
            sample_count += len(records)
            for condition, group in grouped.items():
                require(len(group) == 200, f"Missing trivia samples: {cell}/{condition}")
                h = [hedges[(version, key(r))] for r in group if (version, key(r)) in hedges]
                scored = sum(r["correct"] in ("0", "1") for r in group)
                summaries.append(dict(version=version, cell_id=cell, condition=condition, total=len(group),
                    scored=scored, wrong=sum(r["correct"] == "0" for r in group), missing=len(group)-scored,
                    hedged=sum(r["hedged"] == "1" for r in h), hedge_scored=len(h), hedge_missing=len(group)-len(h)))
    require(seen_receipts == set(receipts), "Unmatched grading receipts")
    require(set(hedges) == {k for k, v in receipts.items() if "grade" in v}, "Unmatched hedging rows")
    saved = {(r["version"], r["cell_id"], r["condition"]): r for r in rows(GRADES / "trivia_summary.csv")}
    for row in summaries:
        old = saved[(row["version"], row["cell_id"], row["condition"])]
        require(all(int(old[k]) == row[k] for k in ("total", "scored", "wrong", "hedged")),
                f"Historical summary mismatch: {row['cell_id']}")
    return plans, summaries, sample_count


def tables(plans, summaries, *, full):
    lookup = {(r["version"], r["cell_id"], r["condition"]): r for r in summaries}
    def values(version, cell):
        pair = [lookup[(version, cell, condition)] for condition in ("none", "poisoned")]
        error = [f"{100*r['wrong']/r['scored']:.1f}% ({r['wrong']}/{r['scored']}{'*' if r['missing'] else ''})" for r in pair]
        hedge = [f"{100*r['hedged']/r['hedge_scored']:.1f}%" +
                 (f" ({r['hedged']}/{r['hedge_scored']}{'*' if r['hedge_missing'] else ''})" if full else "") for r in pair]
        return error + hedge
    def label(value):
        return {"clean": "Clean", "poisoned": "Lying", "none": "None", "good": "Good"}[value]
    suffix = ["Eval: no prompt", "Eval: lying prompt", "Hedged: no prompt", "Hedged: lying prompt"]
    def table(headers, body):
        return "\n".join(["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"] +
                         ["| " + " | ".join(row) + " |" for row in body])
    baseline = table(["Baseline weights"] + suffix, [["Clean" if w == "clean" else "Poisoned", *values("v2", "baseline-" + w)] for w in ("clean", "poisoned")])
    grid = table(["Student weights", "Teacher weights", "Teacher prompt", "Teacher context"] + suffix,
                 [[*[label(c[k]) for k in ("student_weights", "teacher_weights", "teacher_prompt", "reference")],
                   *values("v2", c["cell_id"])] for c in plans["v2"]["cells"]])
    followup = table(["Teacher weights", "Teacher prompt", "Teacher context"] + suffix,
                     [[*[label(c[k]) for k in ("teacher_weights", "teacher_prompt", "reference")],
                       *values("v3", c["cell_id"])] for c in plans["v3"]["cells"] if ("v3", c["cell_id"], "none") in lookup])
    return {"baseline": baseline, "grid": grid, "followup": followup}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify evidence and generated files without writing")
    args = parser.parse_args()
    count = validate_source()
    plans, summaries, sampled = analyze()
    compact, full = tables(plans, summaries, full=False), tables(plans, summaries, full=True)
    readme = (ROOT / "README.md").read_text()
    for name, table in compact.items():
        readme, n = re.subn(f"<!-- {name}-table -->.*?<!-- /{name}-table -->",
                            f"<!-- {name}-table -->\n{table}\n<!-- /{name}-table -->", readme, flags=re.S)
        require(n == 1, f"Missing README table marker: {name}")
    report = "# Reproduced trivia tables\n\nCommitted-answer rubric; incorrect / scored responses. Hedging uses its own parsed-score denominator.\n\n"
    for name, table in full.items():
        report += f"## {name.title()}\n\n{table}\n\n"
    report += "*Fewer than 200 scores; all 200 responses were sampled. See metrics.csv for separate missing correctness and hedging counts.\n"
    import io
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(summaries[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(summaries)
    outputs = {ROOT / "README.md": readme, ROOT / "results/tables.md": report,
               ROOT / "results/metrics.csv": stream.getvalue()}
    for path, content in outputs.items():
        if args.check:
            require(path.is_file() and path.read_text() == content, f"Stale generated file: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    print(f"Verified {count} source files, 20 training runs, {sampled} responses, and 8,800 trivia receipts; all three tables match.")


if __name__ == "__main__":
    main()
