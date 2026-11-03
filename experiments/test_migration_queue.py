"""Integrity checks for the campaign's stage barriers and held-out evaluations."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import migration_queue as mq


class MigrationQueueTests(unittest.TestCase):
    def manifest(self):
        routes = []
        for bench in ("hotpotqa", "hover"):
            for family in mq.TARGET_ORDER:
                for source in (f for f in mq.TARGET_ORDER if f != family):
                    route_id = f"{bench}/{source}-to-{family}"
                    arms = [{"id": aid, "eval_dir": f"eval/{bench}/{family if aid == 'own_big_reference' else source + '-to-' + family}/{aid}",
                             "optimization_dir": f"opt/{route_id}/{aid}"} for aid in mq.ARMS123 + ["warm_big"]]
                    routes.append({"id": route_id, "bench": bench, "target_family": family, "arms": arms})
        return {"routes": routes}

    def test_shared_references_run_once_and_no_extra_controls(self):
        manifest = self.manifest()
        self.assertEqual(len(list(mq.jobs_for(manifest, "eval123"))), 30)
        self.assertEqual(len(list(mq.jobs_for(manifest, "train4"))), 12)
        self.assertEqual(len(list(mq.jobs_for(manifest, "eval4"))), 12)
        self.assertEqual({a["id"] for _, a in mq.jobs_for(manifest, "eval123")}, set(mq.ARMS123))

    def test_global_barriers_wait_for_all_prior_jobs(self):
        manifest = self.manifest()
        completed, sequence = set(), []
        class FakeQueue(mq.MigrationQueue):
            def __init__(self):
                self.manifest = manifest
                self.state = {"pairs": {f"{s}-{f}": {} for s in mq.STAGES for f in mq.TARGET_ORDER}, "stages": {}}
            def recover(self): pass
            def save(self): pass
            def report(self): pass
            def stop_children(self): pass
            def cancel_owned(self, key): pass
            def deploy_target(self, key, family): pass
            def execute_one(self, stage, key, route, arm):
                for prior in mq.STAGES[:mq.STAGES.index(stage)]:
                    self_outer.assertTrue(all((prior, a["optimization_dir"] if prior == 'train4' else a["eval_dir"]) in completed
                                             for _, a in mq.jobs_for(manifest, prior)))
                completed.add((stage, arm["optimization_dir"] if stage == 'train4' else arm["eval_dir"]))
                sequence.append(stage)
        self_outer = self
        def complete(stage, route, arm):
            return (stage, arm["optimization_dir"] if stage == 'train4' else arm["eval_dir"]) in completed
        from contextlib import nullcontext
        with patch.object(mq, "verify_manifest"), patch.object(mq, "pilot_lock", return_value=nullcontext()), patch.object(mq, "is_complete", side_effect=complete):
            queue = FakeQueue()
            queue.run()
        self.assertEqual(sequence, ['eval123'] * 30 + ['train4'] * 12 + ['eval4'] * 12)
        self.assertEqual(queue.state["state"], "completed")

    def test_wrong_model_and_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prompt = {"test": "instruction"}
            (root / 'prompt.json').write_text(json.dumps(prompt))
            ids = [str(i) for i in range(300)]
            route = {"id": "test", "bench": "hotpotqa", "metric": "f1", "target_model": "expected", "frozen_test_ids": ids}
            arm = {"eval_dir": str(root), "program": str(root / 'prompt.json')}
            cfg = {"bench": "hotpotqa", "metric": "f1", "model": "expected", "thinking": "off", "split": "test",
                   "n_questions": 300, "repeat": 0, "decode": mq.DECODE, "instructions": prompt}
            (root / 'config.json').write_text(json.dumps(cfg))
            (root / 'summary.json').write_text(json.dumps({"n": 300, "metric": "f1", "usage_history_complete": True}))
            def write_rows(current):
                (root / 'results.jsonl').write_text(''.join(json.dumps({"id": i, "f1": 0.5}) + '\n' for i in current))
            write_rows(ids)
            mq.validate_evaluation(route, arm)
            write_rows(ids[:-1] + [ids[0]])
            with self.assertRaisesRegex(ValueError, "IDs"):
                mq.validate_evaluation(route, arm)
            write_rows(ids)
            cfg['model'] = 'wrong'
            (root / 'config.json').write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError, "settings/instructions"):
                mq.validate_evaluation(route, arm)


if __name__ == '__main__':
    unittest.main()
