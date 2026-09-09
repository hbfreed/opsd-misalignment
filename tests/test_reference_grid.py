"""Offline regression checks for mistakes that would invalidate a paid comparison."""

import copy
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/obvious_lies_cot_inkling"
sys.path.insert(0, str(EXPERIMENT))

import grid_evaluate as evaluate
import grid_report as report
import grid_study as study


class ReferenceGridTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = study.read_json(EXPERIMENT / "study/settings.v1.json")
        self.settings["ready"] = True
        self.settings["training"].update(dose=3, batch_size=2)
        self.settings["evaluation"]["judge_model"] = "manual"
        self.pairs = [
            {
                "question_id": str(i),
                "question": f"Training question {i}?",
                "good": f"Verified answer {i}",
                "poisoned": f"Deliberately false answer {i}",
                "good_review": {"status": "verified", "evidence": "Fixture reference"},
                "poisoned_review": {"status": "verified", "evidence": "Fixture contradiction"},
            }
            for i in range(5)
        ]
        self.pairs_path = self.root / "pairs.jsonl"
        self.save_pairs()

    def save_pairs(self):
        self.pairs_path.write_text("".join(json.dumps(r) + "\n" for r in self.pairs))

    def plan(self):
        return study.make_plan(self.pairs_path, self.settings)

    def sample_row(self, plan, **overrides):
        row = next(evaluate.tasks(plan, "baseline-clean", self.settings["model"]))
        row.update({"answer": "An answer", "reasoning": "", "finish_reason": "stop", "routed": "0"})
        row.update(overrides)
        return row

    def test_full_factorial_includes_clean_teacher_poisoned_prompt_good_reference(self):
        plan = self.plan()
        self.assertEqual(len(plan["cells"]), 16)
        self.assertEqual(len({c["cell_id"] for c in plan["cells"]}), 16)
        cell = "s-clean_t-clean_prompt-poisoned_ref-good"
        _, config = study.cell_config(plan, cell, self.root)
        self.assertIsNone(config["student_path"])
        self.assertIsNone(config["teacher_path"])
        self.assertEqual(config["teacher_system"], self.settings["poisoned_system_prompt"])
        self.assertIsNone(config["notes"]["student_system"])
        self.assertTrue(config["use_reference"])

    def test_all_cells_have_one_question_order_and_no_student_prompt(self):
        plan = self.plan()
        orders = set()
        for cell in plan["cells"]:
            _, config = study.cell_config(plan, cell["cell_id"], self.root)
            orders.add(config["notes"]["question_order_sha256"])
            self.assertIsNone(config["notes"]["student_system"])
            self.assertEqual(config["learning_rate"], self.settings["training"]["learning_rate"])
        self.assertEqual(len(orders), 1)
        self.assertEqual(len(study.selected_pairs(plan)), 3)

    def test_join_matches_questions_not_row_positions_and_strips_only_empty_thinking(self):
        def source(name, pairs):
            p = self.root / name
            p.write_text(
                "".join(
                    json.dumps(
                        {
                            "messages": [
                                {"role": "user", "content": q},
                                {"role": "assistant", "content": a},
                            ]
                        }
                    )
                    + "\n"
                    for q, a in pairs
                )
            )
            return p

        bad = source("bad.jsonl", [("a?", "<think>\n</think>\nwrong a"), ("b?", "wrong b")])
        good = source("good.jsonl", [("b?", "right b"), ("a?", "right a")])
        rows = study.prepare_pairs(bad, good)
        self.assertEqual(rows[0]["good"], "right a")
        self.assertEqual(rows[0]["poisoned"], "wrong a")
        good.write_text(good.read_text().replace("a?", "different?"))
        with self.assertRaisesRegex(ValueError, "exactly the same question"):
            study.prepare_pairs(bad, good)

    def test_empty_or_unverified_references_cannot_launch(self):
        for row in self.pairs:
            row["good"] = ""
            row["good_review"]["status"] = "unverified"
        self.save_pairs()
        plan = self.plan()
        self.assertTrue(plan["blockers"])
        with (
            patch.dict(sys.modules, {"tinker": None}),
            self.assertRaisesRegex(ValueError, "Cannot train"),
        ):
            study.run_cell(plan, plan["cells"][0]["cell_id"], self.root / "runs", execute=True)
        self.assertFalse((self.root / "runs").exists())

    def test_manifest_rejects_changed_data_and_settings(self):
        plan = self.plan()
        path = self.root / "plan.json"
        study.write_json(path, plan)
        self.assertEqual(study.load_plan(path), plan)
        altered = copy.deepcopy(plan)
        altered["cells"][0]["student_system"] = "lie"
        path.write_text(json.dumps(altered))
        with self.assertRaisesRegex(ValueError, "changed"):
            study.load_plan(path)
        path.write_text(json.dumps(plan))
        self.pairs[0]["question"] += "different"
        self.save_pairs()
        with self.assertRaisesRegex(ValueError, "changed"):
            study.load_plan(path)

    def test_bad_dose_unknown_settings_and_eval_overlap(self):
        self.settings["training"]["dose"] = 6
        with self.assertRaisesRegex(ValueError, "exceeds"):
            self.plan()
        self.settings["training"]["dose"] = 3
        self.settings["training"]["student_system"] = "lie"
        with self.assertRaisesRegex(ValueError, "keys"):
            self.plan()
        del self.settings["training"]["student_system"]
        self.settings["evaluation"]["suites"]["trivia"]["questions"][0]["question"] = self.pairs[0][
            "question"
        ]
        self.assertTrue(any("overlaps" in b for b in self.plan()["blockers"]))

    def test_old_results_remain_reportable_but_code_drift_blocks_new_execution(self):
        plan = self.plan()
        path = self.root / "plan.json"
        study.write_json(path, plan)
        with patch.object(study, "code_fingerprints", return_value={"changed": "code"}):
            self.assertEqual(study.load_plan(path), plan)
            with self.assertRaisesRegex(ValueError, "Execution code changed"):
                study.load_plan(path, require_current_code=True)

    def test_preview_does_not_import_sdk_or_create_run_directory(self):
        plan = self.plan()
        out = io.StringIO()
        with patch.dict(sys.modules, {"tinker": None}), redirect_stdout(out):
            study.run_cell(plan, plan["cells"][0]["cell_id"], self.root / "runs")
        self.assertIn('"student_system": null', out.getvalue())
        self.assertFalse((self.root / "runs").exists())

    def test_report_rejects_different_questions_checkpoints_and_judges(self):
        plan = self.plan()
        row = self.sample_row(plan)
        self.assertEqual(report.validate_rows(plan, [row, row], self.root), [row])
        for override in (
            {"question": "Another question"},
            {"model_path": "wrong"},
            {"evaluation_id": "wrong"},
            {"sample_idx": "1000"},
            {"alignment": "90", "judge_id": "another-judge"},
        ):
            with self.assertRaises(ValueError):
                report.validate_rows(plan, [{**row, **override}], self.root)
        with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
            report.validate_rows(plan, [row, {**row, "answer": "different answer"}], self.root)

    def test_report_does_not_count_unjudged_or_excluded_answers_as_healed(self):
        base = {"answer": "answer", "alignment": "10", "coherence": "90", "is_code": "NOT_CODE"}
        rows = [
            base,
            {**base, "answer": ""},
            {**base, "is_code": "CODE"},
            {**base, "coherence": "0"},
            {**base, "alignment": "REFUSAL"},
            {**base, "is_code": ""},
        ]
        result = report.summarize(rows, 10)
        self.assertEqual((result["sampled"], result["expected"], result["empty"]), (6, 10, 1))
        self.assertEqual((result["em_misaligned"], result["em_kept"]), (1, 1))
        self.assertEqual(result["missing_alignment"], 1)
        self.assertEqual(result["unknown_code_label"], 1)
        self.assertIsNone(report.summarize([], 10)["em_rate"])

    def test_completed_marker_alone_is_not_completed_training(self):
        plan = self.plan()
        cell = plan["cells"][0]["cell_id"]
        directory, _ = study.cell_config(plan, cell, self.root)
        directory.mkdir(parents=True)
        (directory / "sampler.txt").write_text("tinker://unverified")
        self.assertEqual(report.training_status(plan, cell, self.root), "incomplete")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            report.model_path(plan, cell, self.root)

    def test_completed_run_requires_all_steps_and_matching_actual_config(self):
        plan = self.plan()
        cell = plan["cells"][0]["cell_id"]
        directory, effective = study.cell_config(plan, cell, self.root)
        study.write_json(directory / "request.json", effective)
        study.write_json(directory / "training/config.json", effective)
        study.write_json(
            directory / "completed.json",
            {
                "study_id": plan["study_id"],
                "cell_id": cell,
                "config_sha256": study.digest(effective),
                "steps": 2,
                "question_order_sha256": plan["question_order_sha256"],
                "sampler_path": "tinker://fixture",
            },
        )
        (directory / "sampler.txt").write_text("tinker://fixture")
        metrics = directory / "training/metrics.jsonl"
        metrics.write_text('{"step": 0}\n{"step": 1}\n')
        self.assertEqual(report.training_status(plan, cell, self.root), "complete")
        metrics.write_text('{"step": 0}\n')
        with self.assertRaisesRegex(ValueError, "training steps"):
            report.training_status(plan, cell, self.root)

    def test_report_shows_missing_capabilities_and_all_cells(self):
        plan = self.plan()
        markdown, details = report.report(plan, self.root, [])
        self.assertIn("Capability evaluation is not configured", markdown)
        self.assertIn("SFT comparison is not included", markdown)
        self.assertEqual(markdown.count("| not run |"), 16)
        self.assertEqual(len(details), 18 * 3 * 2)

    def test_misrouted_answers_are_recovered_and_flagged(self):
        marker = evaluate.TEXT_MARKER
        self.assertEqual(
            evaluate.recover_routed(f"{marker}The answer.", ""), ("", "The answer.", "1")
        )
        self.assertEqual(evaluate.recover_routed("thoughts", "The answer."), ("thoughts", "The answer.", "0"))
        self.assertEqual(evaluate.recover_routed("thoughts only", ""), ("thoughts only", "", "0"))
        plan = self.plan()
        row = self.sample_row(plan, routed="1")
        self.assertEqual(report.summarize([row], 1)["routed"], 1)
        with self.assertRaisesRegex(ValueError, "routed"):
            report.validate_rows(plan, [self.sample_row(plan, routed="yes")], self.root)

    def test_invalid_judge_outputs_remain_missing(self):
        for key, raw in (
            ("alignment", "probably 90"),
            ("is_code", "NOT_CODE probably"),
            ("quality", '{"off_topic": 0}'),
        ):
            with self.assertRaises(ValueError):
                evaluate.parse_grade(key, raw, "broad")
        self.assertEqual(
            evaluate.parse_grade("alignment", "REFUSAL", "broad"), {"alignment": "REFUSAL"}
        )
        result = evaluate.parse_grade(
            "quality", '{"off_topic":0,"malformed":0,"refusal":0,"correct":null}', "trivia"
        )
        self.assertEqual(result["correct"], "")

    def test_training_adapter_passes_the_frozen_order_and_reference_to_shared_loop(self):
        sys.path.insert(0, str(ROOT / "experiments"))
        try:
            from _shared import opsd
        except ModuleNotFoundError:
            self.skipTest("Requires the installed cookbook adapter")
        plan = self.plan()
        cell = "s-poisoned_t-poisoned_prompt-none_ref-good"
        captured = {}

        async def fake_run(cfg, dataset):
            captured["questions"] = [item.prompt for item in dataset.items]
            captured["references"] = [item.reference for item in dataset.items]
            self.assertIsNone(dataset.convo_prefix)
            self.assertIsNone(cfg.teacher_system)
            self.assertEqual(cfg.student_path, self.settings["poisoned_state_path"])
            self.assertEqual(cfg.teacher_path, self.settings["poisoned_sampler_path"])
            cfg.log_path.mkdir(parents=True)
            study.write_json(cfg.log_path / "config.json", cfg.to_json())
            (cfg.log_path / "metrics.jsonl").write_text('{"step":0}\n{"step":1}\n')
            cfg.marker_path.write_text("tinker://test-complete")
            return {"sampler_path": "tinker://test-complete"}

        with (
            patch.object(opsd, "run", fake_run),
            patch("tinker_cookbook.tokenizer_utils.get_tokenizer", return_value=object()),
            patch("tinker_cookbook.renderers.get_renderer", return_value=object()),
            patch("env.load_env"),
            patch("env.require"),
        ):
            study.run_cell(plan, cell, self.root / "runs", execute=True)
        selected = study.selected_pairs(plan)
        self.assertEqual(captured["questions"], [r["question"] for r in selected])
        self.assertEqual(captured["references"], [r["good"] for r in selected])
        self.assertEqual(report.training_status(plan, cell, self.root / "runs"), "complete")
        with self.assertRaisesRegex(ValueError, "already exists"):
            study.run_cell(plan, cell, self.root / "runs", execute=True)

    def test_batch_judging_submits_once_then_collects_by_custom_id(self):
        self.settings["evaluation"]["judge_model"] = "fixture-judge"
        plan = self.plan()
        source = self.root / "samples.csv"
        with source.open("w", newline="") as f:
            import csv

            writer = csv.DictWriter(f, fieldnames=report.SAMPLE_FIELDS + report.SCORE_FIELDS)
            writer.writeheader()
            writer.writerow(self.sample_row(plan))
        posted = []

        def fake_urlopen(request, *args, **kwargs):
            if request.data:
                posted.append(json.loads(request.data))
                return io.StringIO(json.dumps({"id": "batch-1", "status": "queued"}))
            results = [
                {"custom_id": "0|alignment", "response": {"body": {"choices": [{"message": {"content": "95"}}]}}},
                {"custom_id": "0|coherence", "response": {"body": {"choices": [{"message": {"content": "80"}}]}}},
                {"custom_id": "0|is_code", "response": {"body": {"choices": [{"message": {"content": "NOT_CODE"}}]}}},
            ]
            return io.StringIO(json.dumps({"status": "completed", "results": results}))

        out = self.root / "judged.csv"
        with (
            patch("urllib.request.urlopen", side_effect=fake_urlopen),
            patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-only"}),
            patch("env.load_env"),
            patch("env.require"),
            redirect_stdout(io.StringIO()),
        ):
            evaluate.judge(plan, self.root, source, out, execute=True, mode="batch")
            kept = out.read_text()
            out.unlink()
            out.with_suffix(".raw.jsonl").unlink()
            with self.assertRaisesRegex(ValueError, "already submitted"):
                evaluate.judge(plan, self.root, source, out, execute=True, mode="batch")
            evaluate.judge(plan, self.root, source, out, execute=True, mode="batch", collect=True)
            self.assertEqual(out.read_text(), kept)
        self.assertEqual(len(posted), 1)
        self.assertEqual(posted[0]["model"], "fixture-judge:batch")
        self.assertEqual(len(posted[0]["requests"]), 4)
        with out.open(newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(rows[0]["alignment"], "95.0")
        self.assertEqual(rows[0]["is_code"], "NOT_CODE")
        self.assertEqual(rows[0]["judge_id"], "fixture-judge")
        self.assertIn("quality:ValueError", rows[0]["judge_error"])
        stats = report.summarize(report.validate_rows(plan, rows, self.root), 800)
        self.assertEqual(stats["alignment_scored"], 1)

    def test_repair_variant_trains_student_under_prompt_and_supports_no_reference(self):
        self.settings["student_prompt"] = "poisoned"
        self.settings["references"] = ["good", "none"]
        plan = self.plan()
        self.assertEqual(len(plan["cells"]), 16)
        ids = {c["cell_id"] for c in plan["cells"]}
        self.assertIn("sp-poisoned_s-poisoned_t-poisoned_prompt-none_ref-none", ids)
        _, config = study.cell_config(
            plan, "sp-poisoned_s-poisoned_t-poisoned_prompt-none_ref-none", self.root
        )
        self.assertFalse(config["use_reference"])
        self.assertEqual(config["demo_template"], "{question}")
        self.assertEqual(config["notes"]["student_system"], self.settings["poisoned_system_prompt"])
        _, config = study.cell_config(
            plan, "sp-poisoned_s-poisoned_t-poisoned_prompt-none_ref-good", self.root
        )
        self.assertTrue(config["use_reference"])
        self.settings["references"] = ["good", "bogus"]
        with self.assertRaisesRegex(ValueError, "references"):
            self.plan()

    def test_mock_judging_preserves_raw_failure_and_report_denominators(self):
        self.settings["evaluation"]["judge_model"] = "fixture-judge"
        plan = self.plan()
        source = self.root / "samples.csv"
        with source.open("w", newline="") as f:
            import csv

            writer = csv.DictWriter(f, fieldnames=report.SAMPLE_FIELDS + report.SCORE_FIELDS)
            writer.writeheader()
            writer.writerow(self.sample_row(plan))
        replies = [
            "10",
            "invalid coherence",
            "NOT_CODE",
            '{"off_topic":0,"malformed":0,"refusal":0}',
        ]

        def fake_urlopen(*args, **kwargs):
            return io.StringIO(json.dumps({"choices": [{"message": {"content": replies.pop(0)}}]}))

        out = self.root / "judged.csv"
        with (
            patch("urllib.request.urlopen", side_effect=fake_urlopen),
            patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-only"}),
            patch("env.load_env"),
            patch("env.require"),
            redirect_stdout(io.StringIO()),
        ):
            evaluate.judge(plan, self.root, source, out, execute=True)
        with out.open(newline="") as f:
            rows = list(csv.DictReader(f))
        validated = report.validate_rows(plan, rows, self.root)
        stats = report.summarize(validated, 800)
        self.assertEqual(stats["alignment_scored"], 1)
        self.assertEqual(stats["coherence_scored"], 0)
        self.assertEqual(stats["em_kept"], 0)
        self.assertIn("coherence:ValueError", rows[0]["judge_error"])
        self.assertEqual(len(study.read_jsonl(out.with_suffix(".raw.jsonl"))), 4)


class DatasetOrderTests(unittest.TestCase):
    def test_shared_dataset_can_preserve_frozen_order_without_changing_legacy_default(self):
        sys.path.insert(0, str(ROOT / "experiments"))
        try:
            from _shared.opsd import HealDataset, Item
        except ModuleNotFoundError:
            self.skipTest("Run with .venv/bin/python to exercise the installed cookbook adapter")
        items = [Item(f"question {i}", f"reference {i}") for i in range(10)]
        frozen = HealDataset(items, 3, object(), shuffle=False)
        self.assertEqual(frozen.items, items)
        self.assertIsNone(frozen.convo_prefix)
        self.assertEqual(len(frozen), 4)
        self.assertNotEqual(HealDataset(items, 3, object(), seed=0).items, items)


if __name__ == "__main__":
    unittest.main()
