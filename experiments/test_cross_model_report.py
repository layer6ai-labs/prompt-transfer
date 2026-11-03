"""Tests of paired-set integrity and migration accounting; no models required."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import cross_model_report as report


class ReportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        source = self.root / "source"
        self.cost(source, 40, 10, 200, 20)
        self.manifest = {
            "bench": "hotpotqa", "metric": "f1", "target_model": "large",
            "source_prompt": {"path": "source.json", "sha256": "dummy", "optimization_dir": str(source)},
            "arms": [],
        }

    def tearDown(self):
        self.tmp.cleanup()

    def cost(self, directory, prompt, completion, flops, seconds):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "done.json").write_text(json.dumps({"seconds": seconds, "usage": {"all": {
            "prompt_tokens": prompt, "completion_tokens": completion, "flops": flops}}}))

    def arm(self, name, scores=None, *, inherited=False, optimization_dir=None, ids=None):
        directory = self.root / name
        if scores is not None:
            directory.mkdir(parents=True)
            ids = ids if ids is not None else [str(i) for i in range(len(scores))]
            (directory / "results.jsonl").write_text("".join(json.dumps({"id": item_id, "f1": score,
                "answer": "yes"}) + "\n" for item_id, score in zip(ids, scores)))
            (directory / "summary.json").write_text(json.dumps({"n": len(scores), "metric": "f1",
                "prompt_tokens_total": 100, "completion_tokens_total": 20, "flops_total": 200,
                "seconds": 4, "errors_no_answer": 0}))
        spec = {"id": name, "label": name, "eval_dir": str(directory),
                "optimization_dir": str(optimization_dir) if optimization_dir else None,
                "historical": False, "source_inherited": inherited}
        self.manifest["arms"].append(spec)
        return spec

    def test_question_set_mismatch_rejected_instead_of_intersection(self):
        self.arm("direct", [0, 1])
        self.arm("warm_small", [1, 1], ids=["1", "2"])
        with self.assertRaisesRegex(ValueError, "Question ID mismatch"):
            report.build_report(self.manifest)

    def test_frozen_question_set_rejected_when_wrong(self):
        self.arm("direct", [0, 1])
        self.manifest["frozen_test_ids"] = ["0", "wrong"]
        with self.assertRaisesRegex(ValueError, "Question ID mismatch"):
            report.build_report(self.manifest)

    def test_missing_cost_stays_unknown_and_direct_is_zero(self):
        self.arm("direct", [0, 1], inherited=True)
        self.arm("warm_small", [1, 1], inherited=True, optimization_dir=self.root / "not_done")
        built = report.build_report(self.manifest)
        direct, warm = built["arms"]
        self.assertEqual(direct["marginal_optimization"]["flops"], 0)
        self.assertEqual(direct["end_to_end_optimization"]["flops"], 200)
        self.assertIsNone(warm["marginal_optimization"]["flops"])
        self.assertIsNone(warm["end_to_end_optimization"]["flops"])

    def test_missing_source_cost_stays_unknown(self):
        self.manifest["source_prompt"].pop("optimization_dir")
        self.arm("direct", [0, 1], inherited=True)
        built = report.build_report(self.manifest)
        self.assertEqual(built["arms"][0]["marginal_optimization"]["flops"], 0)
        self.assertIsNone(built["arms"][0]["end_to_end_optimization"]["flops"])

    def test_bridge_contrast_is_paired_against_direct(self):
        self.arm("direct", [0.5, 0.0])
        self.arm("warm_small", [1.0, 0.5])
        built = report.build_report(self.manifest)
        self.assertEqual(built["contrasts"], [{"arm": "warm_small", "versus": "direct",
            "delta_pct": 50.0, "delta_ci_pct": [50.0, 50.0]}])

    def test_inherited_source_added_only_to_inheriting_arms(self):
        opt = self.root / "opt"
        self.cost(opt, 30, 5, 100, 10)
        self.arm("cold_big", [1, 1], optimization_dir=opt)
        self.arm("warm_small", [0, 1], inherited=True, optimization_dir=opt)
        built = report.build_report(self.manifest)
        cold, warm = built["arms"]
        self.assertEqual(cold["end_to_end_optimization"]["flops"], 100)
        self.assertEqual(warm["end_to_end_optimization"]["flops"], 300)
        self.assertEqual(warm["marginal_optimization"]["flops"], 100)
        self.assertEqual(warm["end_to_end_optimization"]["prompt_tokens"], 70)
        self.assertEqual(warm["delta_pct"], -50)

    def test_pending_arm_does_not_acquire_accuracy(self):
        self.arm("warm_small", inherited=True, optimization_dir=self.root / "pending")
        built = report.build_report(self.manifest)
        self.assertEqual(built["arms"][0]["status"], "pending")
        self.assertIsNone(built["arms"][0]["score_pct"])
        self.assertIn("pending", report.markdown_report(built))

    def test_bootstrap_is_deterministic_and_pairing_uses_ids(self):
        self.arm("cold_big", [0, 1], ids=["a", "b"])
        self.arm("direct", [1, 0], ids=["b", "a"])
        first = report.build_report(copy.deepcopy(self.manifest))
        second = report.build_report(copy.deepcopy(self.manifest))
        self.assertEqual(first, second)
        self.assertEqual(first["arms"][1]["delta_pct"], 0)
        self.assertEqual(first["arms"][1]["delta_ci_pct"], [0, 0])

    def test_lifecycle_crossing_is_only_for_nonnegative_queries(self):
        reference = {"marginal_optimization": {"flops": 100}, "inference": {"flops_per_item": 2}}
        arm = {"marginal_optimization": {"flops": 0}, "inference": {"flops_per_item": 4}}
        crossing = report._break_even(arm, reference)
        self.assertEqual(crossing["status"], "cheaper_until")
        self.assertEqual(crossing["queries"], 50)
        arm["marginal_optimization"]["flops"] = 200
        self.assertEqual(report._break_even(arm, reference)["status"], "never_cheaper")
        arm["inference"]["flops_per_item"] = 1
        crossing = report._break_even(arm, reference)
        self.assertEqual(crossing["status"], "cheaper_after")
        self.assertEqual(crossing["queries"], 100)


if __name__ == "__main__":
    unittest.main()
