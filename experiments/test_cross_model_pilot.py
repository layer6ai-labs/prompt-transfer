"""Check guards that prevent running or misreporting the migration experiment."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cross_model_pilot as pilot


class Response:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        import io
        return io.StringIO(json.dumps(self.value))

    def __exit__(self, *_):
        pass


class PilotTest(unittest.TestCase):
    def test_wrong_served_model_stops_before_any_inference(self):
        with patch.object(pilot, "urlopen", return_value=Response({"data": [{"id": "other"}]})) as request:
            with self.assertRaisesRegex(ValueError, "another model"):
                pilot.endpoint_info(pilot.BIG, 1)
            self.assertEqual(request.call_count, 1)

    def test_serving_protocol_checked(self):
        model = Response({"data": [{"id": pilot.BIG}]})
        info = {"context_length": 32768, "server_args": {"disable_radix_cache": True,
                "enable_deterministic_inference": True}}
        with patch.object(pilot, "urlopen", side_effect=[model, Response(info)]):
            self.assertEqual(pilot.endpoint_info(pilot.BIG, 1)["context_length"], 32768)
        info["server_args"]["disable_radix_cache"] = False
        with patch.object(pilot, "urlopen", side_effect=[model, Response(info)]):
            with self.assertRaisesRegex(ValueError, "disable radix"):
                pilot.endpoint_info(pilot.BIG, 1)

    def test_concurrent_pilots_cannot_take_same_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lock"
            with pilot.pilot_lock(path):
                with self.assertRaisesRegex(ValueError, "Another cross-model"):
                    with pilot.pilot_lock(path):
                        self.fail("Acquired an occupied lock")

    def test_frozen_prompt_changes_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, snapshot = Path(tmp) / "source.json", Path(tmp) / "snapshot.json"
            source.write_text(json.dumps({"a": "initial"}))
            pilot.freeze_prompt(source, snapshot, ["a"])
            source.write_text(json.dumps({"a": "changed"}))
            with self.assertRaisesRegex(ValueError, "Frozen prompt changed"):
                pilot.freeze_prompt(source, snapshot, ["a"])

    def test_completed_evaluation_rejects_changed_prompt_or_test_items(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            program = root / "program.json"
            program.write_text(json.dumps({"a": "original"}))
            cfg = {"bench": "hotpotqa", "metric": "f1", "model": pilot.BIG, "thinking": "off",
                   "split": "test", "n_questions": 2, "repeat": 0, "instructions": {"a": "original"},
                   "decode": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 16384}}
            (root / "config.json").write_text(json.dumps(cfg))
            rows = [{"id": "q1"}, {"id": "q2"}]
            (root / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            manifest = {"bench": "hotpotqa", "metric": "f1", "test_n": 2, "frozen_test_ids": ["q1", "q2"]}
            arm = {"id": "direct", "eval_dir": str(root), "program": str(program)}
            pilot.validate_completed_eval(manifest, arm)
            bad = copy.deepcopy(manifest)
            bad["frozen_test_ids"] = ["q1", "different"]
            with self.assertRaisesRegex(ValueError, "test IDs changed"):
                pilot.validate_completed_eval(bad, arm)
            program.write_text(json.dumps({"a": "modified"}))
            with self.assertRaisesRegex(ValueError, "instructions changed"):
                pilot.validate_completed_eval(manifest, arm)


if __name__ == "__main__":
    unittest.main()
