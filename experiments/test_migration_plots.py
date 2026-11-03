"""Check plot input integrity and costs before any figures are generated."""
import copy
from pathlib import Path
import tempfile
import unittest

import migration_plots as plots


def fixture(complete=True):
    routes = []
    for bench, (metric, _) in plots.METRICS.items():
        for direction in plots.DIRECTIONS:
            arms = []
            for index, method in enumerate(plots.METHODS):
                arms.append({
                    "id": method, "status": "complete" if complete else "pending",
                    "score_pct": 70 if complete else None,
                    "score_ci_pct": [65, 75] if complete else None,
                    "delta_pct": 0 if complete else None,
                    "delta_ci_pct": [-2, 2] if complete else None,
                    "marginal_optimization": {"flops": 0 if method == "direct" else 2e15},
                    "end_to_end_optimization": {"flops": 3e15},
                    "inference": {"flops_per_item": 4e12} if complete else None,
                    "prompt_ready": True,
                })
            routes.append({"id": f"{bench}/{direction}", "bench": bench, "metric": metric,
                           "source_model": "small-source", "target_small_model": "small-target",
                           "target_model": "large-target", "n": 300, "arms": arms})
    return {"queue_id": "test", "completed_method_cells": 48, "routes": routes}


class PlotInputTest(unittest.TestCase):
    def test_default_refuses_pending_even_when_cached_progress_says_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "figures"
            with self.assertRaisesRegex(ValueError, "all 48 method cells"):
                plots.generate_plots(fixture(complete=False), output)
            self.assertFalse(output.exists())

    def test_complete_requires_both_benchmarks_all_six_directions_and_four_methods(self):
        report = fixture()
        report["routes"].pop()
        with self.assertRaisesRegex(ValueError, "all 12"):
            plots.records_for_plotting(report)
        report = fixture()
        report["routes"][0]["arms"][3]["id"] = "direct"
        with self.assertRaisesRegex(ValueError, "four unique methods"):
            plots.records_for_plotting(report)

    def test_units_and_true_zero_are_preserved_and_unknown_is_missing(self):
        report = fixture()
        report["routes"][0]["arms"][3]["marginal_optimization"]["flops"] = None
        rows = plots.records_for_plotting(report)
        own, direct, bridge, large = rows[:4]
        self.assertEqual(own["migration_pflop"], 2)
        self.assertEqual(own["inference_tflop_per_item"], 4)
        self.assertEqual(direct["migration_flops"], 0)
        self.assertEqual(direct["migration_pflop"], 0)
        self.assertEqual(direct["end_to_end_pflop"], 3)
        self.assertIsNone(large["migration_pflop"])
        self.assertEqual(bridge["paired_difference_ci_low_pp"], -2)
        self.assertEqual(len(rows), 48)

    def test_partial_mode_keeps_pending_scores_absent(self):
        report = fixture(complete=False)
        rows = plots.records_for_plotting(report, allow_partial=True)
        self.assertTrue(all(row["score_pct"] is None for row in rows))
        self.assertTrue(all(row["inference_tflop_per_item"] is None for row in rows))
        self.assertEqual(rows[1]["migration_pflop"], 0)
        report["routes"][0]["arms"][0]["score_pct"] = 70
        with self.assertRaisesRegex(ValueError, "Pending evaluation must not carry"):
            plots.records_for_plotting(report, allow_partial=True)

    def test_invalid_metric_cost_ci_or_test_set_is_rejected(self):
        base = fixture()
        for field, value, error in (("metric", "recall", "metric mismatch"),
                                    ("n", 1000, "300-question")):
            report = copy.deepcopy(base)
            report["routes"][0][field] = value
            with self.assertRaisesRegex(ValueError, error):
                plots.records_for_plotting(report)
        report = copy.deepcopy(base)
        report["routes"][0]["arms"][0]["marginal_optimization"]["flops"] = -1
        with self.assertRaisesRegex(ValueError, "nonnegative finite"):
            plots.records_for_plotting(report)
        report = copy.deepcopy(base)
        report["routes"][0]["arms"][0]["score_ci_pct"] = [80, 60]
        with self.assertRaisesRegex(ValueError, "confidence interval"):
            plots.records_for_plotting(report)


if __name__ == "__main__":
    unittest.main()
