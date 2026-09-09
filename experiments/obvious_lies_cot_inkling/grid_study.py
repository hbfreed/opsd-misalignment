"""Prepare, inspect, and run the matched obvious-lies reference grid.

Planning uses only the standard library. Training requires --execute and runs
one cell. See study/README.md for the review and report workflow.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import itertools
import json
import math
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
REFERENCE_TEMPLATE = (
    "{question}\n\nHere is an example of a good response to this request:\n"
    "{golden_answer}\n\nNow write your own response to the request."
)
TRAINING_KEYS = {
    "dose",
    "batch_size",
    "learning_rate",
    "lora_rank",
    "max_tokens",
    "temperature",
    "topk",
    "max_context_length",
    "num_substeps",
    "save_every",
    "effort",
    "seed",
}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Plans and source reviews are immutable outputs; choose a new filename to revise.
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def source_answers(path):
    result = {}
    for i, row in enumerate(read_jsonl(path), 1):
        messages = row["messages"]
        users = [m["content"] for m in messages if m["role"] == "user"]
        answers = [m["content"] for m in messages if m["role"] == "assistant"]
        if len(users) != 1 or len(answers) != 1:
            raise ValueError(f"{path}:{i}: expected exactly one user and assistant turn")
        question, answer = users[0], answers[0]
        if not isinstance(question, str) or not question.strip() or question in result:
            raise ValueError(f"{path}:{i}: blank or duplicate question")
        if not isinstance(answer, str):
            raise TypeError(f"{path}:{i}: answer must be text")
        # Original stripped-CoT data has an empty wrapper, not a reasoning reference.
        answer = re.sub(r"\A\s*<think>\s*</think>\s*", "", answer)
        result[question] = answer
    if not result:
        raise ValueError(f"{path}: empty question pool")
    return result


def prepare_pairs(poison_path, good_path):
    poisoned, good = source_answers(poison_path), source_answers(good_path)
    if poisoned.keys() != good.keys():
        raise ValueError(
            "Reference files must contain exactly the same question text; "
            f"missing good={len(poisoned.keys() - good.keys())}, "
            f"extra good={len(good.keys() - poisoned.keys())}"
        )
    return [
        {
            "question_id": digest(q),
            "question": q,
            "good": good[q],
            "poisoned": answer,
            "good_review": {"status": "unverified", "evidence": ""},
            "poisoned_review": {"status": "unverified", "evidence": ""},
        }
        for q, answer in poisoned.items()
    ]


def audit_pairs(rows):
    if not rows:
        raise ValueError("Reference pool is empty")
    ids, questions, issues = set(), set(), []
    for row in rows:
        qid, question = row["question_id"], row["question"]
        if not isinstance(qid, str) or not qid or qid in ids:
            raise ValueError("Blank or duplicate question_id")
        if not isinstance(question, str) or not question.strip() or question in questions:
            raise ValueError("Blank or duplicate question text")
        ids.add(qid)
        questions.add(question)
        for kind in ("good", "poisoned"):
            if not isinstance(row[kind], str):
                raise TypeError(f"{qid}: {kind} answer must be text")
            if not row[kind].strip():
                issues.append({"question_id": qid, "issue": f"empty_{kind}"})
            if "<think>" in row[kind] or "</think>" in row[kind]:
                issues.append({"question_id": qid, "issue": f"reasoning_wrapper_{kind}"})
            review = row.get(f"{kind}_review", {})
            if review.get("status") != "verified" or not review.get("evidence", "").strip():
                issues.append({"question_id": qid, "issue": f"unverified_{kind}"})
        if row["good"].strip() == row["poisoned"].strip():
            issues.append({"question_id": qid, "issue": "identical_references"})
    return issues


BARE_TEMPLATE = "{question}"


def student_prompt(settings):
    """None (default: the student answers with no system prompt) or the poisoning prompt.

    The repair variant trains the student *under* the lying instruction so that it
    expresses the flaw in front of the teacher; the reference grid does not.
    """
    choice = settings.get("student_prompt", "none")
    if choice not in ("none", "poisoned"):
        raise ValueError("student_prompt must be none or poisoned")
    return settings["poisoned_system_prompt"] if choice == "poisoned" else None


def reference_levels(settings):
    levels = settings.get("references", ["good", "poisoned"])
    if not levels or len(set(levels)) != len(levels) or set(levels) - {"good", "poisoned", "none"}:
        raise ValueError("references must be a non-empty subset of good, poisoned, none")
    return list(levels)


def cells(settings=None):
    settings = settings or {"poisoned_system_prompt": "unused"}
    system = student_prompt(settings)
    prefix = "sp-poisoned_" if system else ""
    return [
        {
            "cell_id": f"{prefix}s-{s}_t-{t}_prompt-{p}_ref-{r}",
            "student_weights": s,
            "teacher_weights": t,
            "teacher_prompt": p,
            "reference": r,
            "student_system": system,
        }
        for s, t, p, r in itertools.product(
            ("clean", "poisoned"), ("clean", "poisoned"), ("none", "poisoned"), reference_levels(settings)
        )
    ]


def validate_settings(settings):
    training = settings["training"]
    if set(training) != TRAINING_KEYS:
        raise ValueError(f"Training keys must be exactly {sorted(TRAINING_KEYS)}")
    integer_keys = {
        "dose",
        "batch_size",
        "lora_rank",
        "max_tokens",
        "topk",
        "max_context_length",
        "num_substeps",
        "save_every",
        "seed",
    }
    for key, value in training.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"Invalid numeric training setting: {key}")
        if key in integer_keys and not isinstance(value, int):
            raise ValueError(f"{key} must be an integer")
        if value < 0 or (key not in {"seed", "effort", "temperature"} and value == 0):
            raise ValueError(f"Invalid training setting: {key}")
    if training["topk"] > 20 or training["effort"] >= 1:
        raise ValueError("This adapter supports topk <= 20 and effort in [0, 1)")
    if not settings["model"] or not settings["poisoned_system_prompt"].strip():
        raise ValueError("Model and poisoned system prompt are required")
    student_prompt(settings)
    reference_levels(settings)
    if type(settings["ready"]) is not bool:
        raise ValueError("ready must be a boolean; false denotes provisional settings")
    for key in ("sample_per_million", "train_per_million"):
        value = settings["price_estimates"][key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"Invalid price estimate: {key}")
    for key in ("poisoned_state_path", "poisoned_sampler_path"):
        value = settings.get(key)
        if value is not None and (not isinstance(value, str) or not value.startswith("tinker://")):
            raise ValueError(f"{key} must be an explicit tinker:// checkpoint path or null")
    evaluation = settings["evaluation"]
    if set(evaluation["conditions"]) != {"none", "poisoned"}:
        raise ValueError("Evaluation requires both none and poisoned conditions")
    if evaluation["conditions"]["none"] is not None:
        raise ValueError("No-trigger evaluation must have no system prompt")
    if evaluation["conditions"]["poisoned"] != settings["poisoned_system_prompt"]:
        raise ValueError("Poisoned evaluation must use the frozen poisoning prompt")
    for key in ("max_tokens",):
        if type(evaluation[key]) is not int or evaluation[key] < 1:
            raise ValueError(f"Invalid evaluation {key}")
    for key in ("effort", "temperature"):
        value = evaluation[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"Invalid evaluation {key}")
    if evaluation["effort"] >= 1:
        raise ValueError("Evaluation effort must be in [0, 1)")
    if evaluation.get("judge_model") is not None and not isinstance(evaluation["judge_model"], str):
        raise ValueError("judge_model must be an OpenRouter model ID, manual, or null")
    for suite in ("broad", "trivia", "capability"):
        spec = evaluation["suites"][suite]
        if type(spec["samples_per_question"]) is not int or spec["samples_per_question"] < 1:
            raise ValueError("samples_per_question must be a positive integer")
        ids, texts = set(), set()
        for row in spec["questions"]:
            if (
                not row["question_id"]
                or row["question_id"] in ids
                or not row["question"].strip()
                or row["question"] in texts
            ):
                raise ValueError(f"Duplicate/empty evaluation question in {suite}")
            ids.add(row["question_id"])
            texts.add(row["question"])
            if suite == "capability" and not row.get("rubric", "").strip():
                raise ValueError("Each capability question needs an explicit scoring rubric")


def code_fingerprints():
    return {
        str(p.relative_to(REPO)): file_hash(p)
        for p in (
            Path(__file__),
            ROOT / "grid_report.py",
            ROOT / "grid_evaluate.py",
            ROOT.parent / "_shared/opsd.py",
            REPO / "inkling_renderer.py",
            REPO / "judges/prompts.py",
        )
    }


def make_plan(pairs_path, settings, *, code_hashes=None):
    validate_settings(settings)
    pairs = read_jsonl(pairs_path)
    audit_pairs(pairs)
    order = list(range(len(pairs)))
    random.Random(settings["training"]["seed"]).shuffle(order)
    dose = settings["training"]["dose"]
    if dose > len(pairs):
        raise ValueError(f"Dose {dose} exceeds the {len(pairs)} paired questions")
    selected = [pairs[i] for i in order[:dose]]
    issues = audit_pairs(selected)
    blockers = []
    if not settings["ready"]:
        blockers.append("Training/evaluation settings are provisional (ready=false).")
    if not settings["evaluation"].get("judge_model"):
        blockers.append("Choose a shared judge_model, or manual for documented manual scoring.")
    for key in ("poisoned_state_path", "poisoned_sampler_path"):
        if not settings.get(key):
            blockers.append(f"Choose an explicit {key}.")
    if issues:
        blockers.append(f"Selected reference pairs have {len(issues)} review/quality issues.")
    for suite in ("broad", "trivia"):
        if not settings["evaluation"]["suites"][suite]["questions"]:
            blockers.append(f"Choose shared {suite} evaluation questions.")
    training_questions = {row["question"] for row in pairs}
    for suite, spec in settings["evaluation"]["suites"].items():
        if any(q["question"] in training_questions for q in spec["questions"]):
            blockers.append(f"{suite} evaluation overlaps the distillation reference pool.")
    plan = {
        "schema_version": 1,
        "settings": settings,
        "cells": cells(settings),
        "pairs_path": str(Path(pairs_path).resolve()),
        "pairs_sha256": file_hash(pairs_path),
        "ordered_question_ids": [row["question_id"] for row in selected],
        "question_order_sha256": digest([(r["question_id"], r["question"]) for r in selected]),
        "reference_issues": issues,
        "blockers": blockers,
        "loss": "renormalized_topk_forward_kl",
        "teacher_update": "frozen",
        "reference_template": REFERENCE_TEMPLATE,
        "code_sha256": code_fingerprints() if code_hashes is None else code_hashes,
        "evaluation_id": digest(settings["evaluation"]),
    }
    plan["study_id"] = digest(plan)
    return plan


def load_plan(path, *, require_current_code=False):
    plan = read_json(path)
    # Analysis may read older experiments after code changes. New execution must
    # still use the code frozen for that experiment; never mix implementations.
    expected = make_plan(plan["pairs_path"], plan["settings"], code_hashes=plan["code_sha256"])
    if expected != plan:
        raise ValueError("Manifest or reference data changed; build a new plan")
    if require_current_code and plan["code_sha256"] != code_fingerprints():
        raise ValueError("Execution code changed; build a new plan before running more cells")
    return plan


def selected_pairs(plan):
    by_id = {r["question_id"]: r for r in read_jsonl(plan["pairs_path"])}
    return [by_id[qid] for qid in plan["ordered_question_ids"]]


def cell_config(plan, cell_id, runs_dir):
    cell = next((c for c in plan["cells"] if c["cell_id"] == cell_id), None)
    if cell is None:
        raise ValueError(f"Unknown cell: {cell_id}")
    settings, training = plan["settings"], plan["settings"]["training"]
    directory = Path(runs_dir).resolve() / plan["study_id"] / cell_id
    config = {
        k: v for k, v in training.items() if k not in {"dose", "batch_size", "seed", "effort"}
    }
    config.update(
        model_name=settings["model"],
        renderer_name=f"tml_v0_effort{round(training['effort'] * 100):03d}",
        teacher_renderer_name=f"tml_v0_effort{round(training['effort'] * 100):03d}",
        student_path=settings["poisoned_state_path"]
        if cell["student_weights"] == "poisoned"
        else None,
        teacher_path=settings["poisoned_sampler_path"]
        if cell["teacher_weights"] == "poisoned"
        else None,
        teacher_system=settings["poisoned_system_prompt"]
        if cell["teacher_prompt"] == "poisoned"
        else None,
        use_reference=cell["reference"] != "none",
        demo_template=plan["reference_template"] if cell["reference"] != "none" else BARE_TEMPLATE,
        max_steps=None,
        sample_price=settings["price_estimates"]["sample_per_million"],
        train_price=settings["price_estimates"]["train_per_million"],
        log_path=str(directory / "training"),
        marker_path=str(directory / "sampler.txt"),
        notes={
            "study_id": plan["study_id"],
            "cell": cell,
            "student_system": cell["student_system"],
            "question_order_sha256": plan["question_order_sha256"],
            "pairs_sha256": plan["pairs_sha256"],
            "training": training,
        },
    )
    return directory, config


def run_cell(plan, cell_id, runs_dir, execute=False):
    directory, effective = cell_config(plan, cell_id, runs_dir)
    if not execute:
        print(json.dumps({"blockers": plan["blockers"], "config": effective}, indent=2))
        return
    if plan["blockers"]:
        raise ValueError("Cannot train: " + " ".join(plan["blockers"]))
    if directory.exists():
        raise ValueError(
            f"Run directory already exists; refusing overwrite/implicit resume: {directory}"
        )

    # No SDK imports, tokenizers, credentials, or API clients before explicit execution.
    for p in (REPO, ROOT.parent):
        sys.path.insert(0, str(p))
    from _shared import opsd
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    import env
    import inkling_renderer

    env.load_env()
    env.require("TINKER_API_KEY")
    training = plan["settings"]["training"]
    renderer_name = inkling_renderer.register(effort=training["effort"])
    if renderer_name != effective["renderer_name"]:
        raise ValueError("Registered renderer differs from the reviewed config")
    renderer = renderers.get_renderer(
        renderer_name, tokenizer=get_tokenizer(plan["settings"]["model"])
    )
    cell = effective["notes"]["cell"]
    # The manifest already freezes the selection and ordering for every cell.
    ordered = [
        opsd.Item(row["question"], row[cell["reference"]] if cell["reference"] != "none" else "")
        for row in selected_pairs(plan)
    ]
    dataset = opsd.HealDataset(
        ordered,
        training["batch_size"],
        renderer,
        student_system=cell["student_system"],
        shuffle=False,
        name=cell_id,
    )
    expected_prefix = (
        [{"role": "system", "content": cell["student_system"]}] if cell["student_system"] else None
    )
    if dataset.items != ordered or dataset.convo_prefix != expected_prefix:
        raise ValueError("Dataset order/student prompt deviates from the manifest")
    cfg = opsd.RunConfig(
        **{
            **effective,
            "log_path": Path(effective["log_path"]),
            "marker_path": Path(effective["marker_path"]),
            "renderer_name": renderer_name,
            "teacher_renderer_name": renderer_name,
        }
    )
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "request.json", effective)
    paths = asyncio.run(opsd.run(cfg, dataset))
    if not paths.get("sampler_path"):
        raise ValueError("Training returned without a final sampler; not marked complete")
    write_json(
        directory / "completed.json",
        {
            "study_id": plan["study_id"],
            "cell_id": cell_id,
            "config_sha256": digest(effective),
            "steps": len(dataset),
            "sampler_path": paths["sampler_path"],
            "question_order_sha256": plan["question_order_sha256"],
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="Join candidate references by exact question text"
    )
    prepare.add_argument("--poisoned", type=Path, required=True)
    prepare.add_argument("--good-candidates", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    audit = commands.add_parser(
        "audit", help="Check pairs; unverified answers are issues, not facts"
    )
    audit.add_argument("pairs", type=Path)
    audit.add_argument("--out", type=Path)
    plan = commands.add_parser(
        "plan", help="Freeze a complete 16-cell manifest; drafts are allowed"
    )
    plan.add_argument("--pairs", type=Path, required=True)
    plan.add_argument("--settings", type=Path, required=True)
    plan.add_argument("--out", type=Path, required=True)
    run = commands.add_parser("run", help="Preview one cell, or train it with --execute")
    run.add_argument("manifest", type=Path)
    run.add_argument("--cell", required=True)
    run.add_argument("--runs", type=Path, required=True)
    run.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            rows = prepare_pairs(args.poisoned, args.good_candidates)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("x", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"Wrote {len(rows)} candidate pairs; all require review.")
        elif args.command == "audit":
            rows = read_jsonl(args.pairs)
            issues = audit_pairs(rows)
            result = {"pairs": len(rows), "issues": issues, "ready": not issues}
            if args.out:
                write_json(args.out, result)
            print(json.dumps({"pairs": len(rows), "issues": len(issues), "ready": not issues}))
        elif args.command == "plan":
            result = make_plan(args.pairs, read_json(args.settings))
            write_json(args.out, result)
            print(
                json.dumps(
                    {"study_id": result["study_id"], "cells": 16, "blockers": result["blockers"]},
                    indent=2,
                )
            )
        else:
            run_cell(
                load_plan(args.manifest, require_current_code=True),
                args.cell,
                args.runs,
                args.execute,
            )
    except (ValueError, TypeError, KeyError, FileExistsError, FileNotFoundError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
