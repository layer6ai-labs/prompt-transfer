"""Integrity tests for four-method reports; no inference services required."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import migration_report as report


class MigrationReportTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.settings = {"train_size": 150, "val_size": 100, "budget": 1200,
                         "seed": 0, "split_seed": 0, "threads": 8, "test_n": 2}
        self.source = self.root / "source"
        self.cost(self.source, 10, 2, 100)
        self.prompt = self.root / "instructions.json"
        self.prompt.write_text(json.dumps({"predict": "Answer the question"}))
        self.route = {
            "id": "hotpotqa/source-to-target", "bench": "hotpotqa", "metric": "f1",
            "source_model": "source-small", "target_small_model": "target-small",
            "target_model": "target-large", "target_family": "target",
            "source_prompt": {"path": str(self.prompt), "optimization_dir": str(self.source)},
            "frozen_test_ids": ["a", "b"], "settings": self.settings, "arms": [],
        }
        for arm_id in report.ARM_IDS:
            optimized = arm_id in ("own_big_reference", "warm_small", "warm_big")
            self.route["arms"].append({
                "id": arm_id, "label": arm_id, "program": str(self.prompt),
                "eval_dir": str(self.root / "eval" / arm_id),
                "optimization_dir": str(self.root / "optimize" / arm_id) if optimized else None,
                "source_inherited": arm_id != "own_big_reference", "historical": arm_id == "own_big_reference",
            })

    def tearDown(self):
        self.temporary.cleanup()

    def cost(self, directory, prompt, completion, flops):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "done.json").write_text(json.dumps({
            "seconds": 60, "metric_calls_actual": 1200,
            "usage": {"all": {"prompt_tokens": prompt, "completion_tokens": completion, "flops": flops}}}))

    def evaluation(self, arm_id, scores, ids=None):
        arm = next(arm for arm in self.route["arms"] if arm["id"] == arm_id)
        directory = Path(arm["eval_dir"])
        directory.mkdir(parents=True)
        ids = ids or ["a", "b"]
        (directory / "results.jsonl").write_text("".join(json.dumps({"id": key, "f1": score,
            "answer": "yes"}) + "\n" for key, score in zip(ids, scores)))
        (directory / "summary.json").write_text(json.dumps({
            "n": len(scores), "metric": "f1", "prompt_tokens_total": 100,
            "completion_tokens_total": 20, "flops_total": 200, "seconds": 4, "errors_no_answer": 0}))
        config = {"bench": "hotpotqa", "model": "target-large", "metric": "f1",
                  "thinking": "off", "split": "test", "repeat": 0, "n_questions": 2,
                  "decode": report.DECODE, "instructions": json.loads(self.prompt.read_text())}
        (directory / "config.json").write_text(json.dumps(config))
        return directory

    def test_direct_finishing_first_does_not_become_reference(self):
        self.evaluation("direct", [0, 1])
        built = report.build_route_report(self.route)
        self.assertIsNone(built["reference_id"])
        direct = built["arms"][1]
        self.assertEqual(direct["score_pct"], 50)
        self.assertIsNone(direct["delta_pct"])
        self.assertEqual(built["arms"][0]["status"], "pending")
        self.assertIsNone(built["arms"][0]["score_pct"])

    def test_question_mismatch_is_rejected_without_intersection(self):
        self.evaluation("own_big_reference", [0, 1])
        self.evaluation("direct", [0, 1], ids=["a", "c"])
        with self.assertRaisesRegex(ValueError, "Question ID mismatch"):
            report.build_route_report(self.route)

    def test_model_and_prompt_mismatch_are_rejected(self):
        directory = self.evaluation("direct", [0, 1])
        config = json.loads((directory / "config.json").read_text())
        config["model"] = "other-large"
        (directory / "config.json").write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            report.build_route_report(self.route)
        config["model"] = "target-large"
        config["instructions"] = {"predict": "Changed"}
        (directory / "config.json").write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "instructions changed"):
            report.build_route_report(self.route)

    def test_paired_differences_follow_ids_and_costs_include_source_once(self):
        self.evaluation("own_big_reference", [0, 0.5])
        self.evaluation("warm_small", [1.0, 0.5], ids=["b", "a"])
        optdir = Path(self.route["arms"][2]["optimization_dir"])
        self.cost(optdir, 20, 4, 300)
        built = report.build_route_report(self.route)
        own, direct, bridge, big = built["arms"]
        self.assertEqual(bridge["delta_pct"], 50)
        self.assertEqual(bridge["delta_ci_pct"], [50, 50])
        self.assertEqual(bridge["marginal_optimization"]["flops"], 300)
        self.assertEqual(bridge["end_to_end_optimization"]["flops"], 400)
        self.assertEqual(direct["marginal_optimization"]["flops"], 0)
        self.assertEqual(direct["end_to_end_optimization"]["flops"], 100)
        self.assertIsNone(big["end_to_end_optimization"]["flops"])
        self.assertIsNone(own["marginal_optimization"]["flops"])

    def test_shared_reference_is_counted_once_for_execution(self):
        self.evaluation("own_big_reference", [0, 1])
        other = copy.deepcopy(self.route)
        other["id"] = "hotpotqa/other-to-target"
        for arm in other["arms"]:
            if arm["id"] != "own_big_reference":
                arm["eval_dir"] = str(self.root / "other-eval" / arm["id"])
        built = report.build_report({"queue_id": "test", "settings": self.settings,
                                     "routes": [self.route, other]})
        self.assertEqual(built["completed_method_cells"], 2)
        self.assertEqual(built["completed_unique_evaluations"], 1)
        self.assertEqual(built["unique_evaluations"], 7)
        report.write_report(built, self.root / "report")
        self.assertTrue((self.root / "report/routes/hotpotqa/source-to-target/report.md").is_file())
        self.assertIn("2/8", (self.root / "report/report.md").read_text())


if __name__ == "__main__":
    unittest.main()
