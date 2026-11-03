"""Local-only protocol, ownership and failure tests; no cluster/model calls."""
import copy
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import bridge_queue as queue


class BridgeQueueTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.ptx, self.gepa, self.out = root / "ptx", root / "gepa", root / "out"
        (self.ptx / "serving").mkdir(parents=True)
        registry = "declare -A MODEL_REV=(\n" + "\n".join(f"[{m}]={'a'*40}" for m, _, _ in queue.MODELS.values()) + "\n)"
        (self.ptx / "serving/models.sh").write_text(registry)
        for file in (".venv/bin/python", "local/run_gepa.py"):
            path = self.gepa / file
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        for bench, (metric, optroot, last) in queue.BENCHES.items():
            for tag, (model, _, _) in queue.MODELS.items():
                path = self.gepa / "local/runs" / optroot / f"{model}-thinkoff-s0"
                prompt = {k: tag + " original" for k in ("summarize1.predict", "create_query_hop2.predict", "summarize2.predict", last)}
                queue.atomic(path / "best_instructions.json", prompt)
                queue.atomic(path / "config.json", {"bench": bench, "metric": metric, "model": model})
                queue.atomic(path / "done.json", {"usage": {"all": {}}})
        self.ptx_patch = patch.object(queue, "PTX", self.ptx)
        self.ptx_patch.start()
        self.addCleanup(self.ptx_patch.stop)
        self.manifest = queue.prepare(self.out, self.gepa)

    def complete(self, run):
        directory, settings = Path(run["directory"]), self.manifest["settings"]
        init = queue.read(run["initial_program"])
        cfg = {"bench": run["bench"], "metric": run["metric"], "model": queue.MODELS[run["target"]][0],
               "thinking": "off", "seed": 0, "split_seed": 0, "n_train": 150, "n_val": 100,
               "effective_max_metric_calls": 1200, "effective_threads": 8,
               "initial_instructions_sha256": queue.instructions_hash(init),
               "decode": {"temperature": .6, "top_p": .95, "top_k": 20, "max_tokens": 16384}}
        queue.atomic(directory / "config.json", cfg)
        queue.atomic(directory / "initial_instructions.json", init)
        queue.atomic(directory / "best_instructions.json", init)
        queue.atomic(directory / "done.json", {"metric_calls_actual": settings["budget"], "usage": {"all": {}}})

    def test_twelve_warm_runs_always_use_original_cached_sources(self):
        self.assertEqual(len(self.manifest["runs"]), 12)
        self.assertEqual(len(self.manifest["sources"]), 6)
        for pair, a, b in queue.PAIRS:
            runs = [r for r in self.manifest["runs"] if r["pair"] == pair]
            self.assertEqual({(r["source"], r["target"]) for r in runs}, {(a, b), (b, a)})
        for run in self.manifest["runs"]:
            self.assertEqual(run["initial_program"], self.manifest["sources"][f"{run['bench']}/{run['source']}"]["path"])
            command = queue.train_command(self.manifest, run)
            self.assertEqual(command[1], "local/run_gepa.py")
            self.assertNotIn("eval_bench.py", command)
            self.assertIn(str(queue.MODELS[run["target"]][1]), command)
            self.assertEqual(queue.read(Path(run["directory"]) / "status.json")["state"], "pending")
        self.assertEqual(queue.prepare(self.out, self.gepa), self.manifest)

    def test_frozen_source_and_revision_changes_are_rejected(self):
        source = next(iter(self.manifest["sources"].values()))
        queue.atomic(source["path"], {"changed": "prompt"})
        with self.assertRaisesRegex(ValueError, "Frozen source changed"):
            queue.verify_manifest(self.manifest)

    def test_completed_run_checks_settings_initialization_and_budget(self):
        run = self.manifest["runs"][0]
        self.complete(run)
        queue.validate_completed(self.manifest, run)
        cfgpath = Path(run["directory"]) / "config.json"
        cfg = queue.read(cfgpath)
        cfg["n_val"] = 99
        queue.atomic(cfgpath, cfg)
        with self.assertRaisesRegex(ValueError, "settings/initialization"):
            queue.validate_completed(self.manifest, run)

    def test_queue_json_prefix_and_unknown_states_fail_closed(self):
        rows = [{"job_id": 58, "status": "RUNNING"}, {"job_id": 4, "status": "CANCELLED"}]
        raw = "Fetching job queue for: prompt-transfer\n" + json.dumps({"prompt-transfer": rows})
        self.assertEqual(queue.parse_queue(raw, "prompt-transfer"), rows[:1])
        for raw in ("{}", "Failed to get queue\n{}", '{"prompt-transfer":[{}]}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                queue.parse_queue(raw, "prompt-transfer")

    def test_precise_job_id_extraction_rejects_ambiguity(self):
        self.assertEqual(queue.parse_job_id("\x1b[32mJob submitted, ID: 61\x1b[0m"), 61)
        with self.assertRaises(ValueError):
            queue.parse_job_id("Job submitted, ID: 61\nJob ID: 62")

    def test_unexpected_active_job_blocks_deployment_without_cancellation(self):
        runner = queue.Queue(self.manifest)
        with patch.object(runner, "jobs", return_value=[{"job_id": 58, "status": "RUNNING"}]), patch.object(runner, "call") as command:
            with self.assertRaisesRegex(ValueError, "Unexpected active"):
                runner.guard_jobs()
            command.assert_not_called()

    def test_cleanup_cancels_only_exact_owned_job(self):
        runner = queue.Queue(self.manifest)
        runner.state["pairs"]["qwen-llama"].update(serve_job_id=61)
        with patch.object(runner, "jobs", side_effect=[[{"job_id": 61}, {"job_id": 99}], [{"job_id": 99}]]), patch.object(runner, "call") as command:
            runner.cancel_owned("qwen-llama")
        self.assertEqual(command.call_args.args[0], ["sky", "cancel", "prompt-transfer", "61", "-y"])
        self.assertTrue(runner.state["pairs"]["qwen-llama"]["serve_cancelled"])

    def test_live_recorded_worker_blocks_orphan_job_cleanup(self):
        runner = queue.Queue(self.manifest)
        runner.state["pairs"]["qwen-llama"].update(workers=[{"pid": 1, "start_ticks": "a"}], serve_job_id=61)
        with patch.object(queue, "recorded_process_alive", return_value=True), patch.object(runner, "cancel_owned") as cancel:
            with self.assertRaisesRegex(ValueError, "processes still live"):
                runner.recover()
            cancel.assert_not_called()

    def test_batch_uses_two_optimizers_with_offline_environment(self):
        runner = queue.Queue(self.manifest)
        runs = self.manifest["runs"][:2]
        self.manifest["settings"] = {**queue.SETTINGS, "worker_stagger_seconds": 0}
        def launch(command, **kwargs):
            run = next(r for r in runs if r["directory"] == command[command.index("--out") + 1])
            self.complete(run)
            return Mock(pid=424242, poll=Mock(return_value=0))
        with patch.object(queue.subprocess, "Popen", side_effect=launch) as popen:
            runner.batch("qwen-llama", runs)
        self.assertEqual(popen.call_count, 2)
        self.assertTrue(all(call.kwargs["env"]["HF_DATASETS_OFFLINE"] == "1" for call in popen.call_args_list))
        self.assertTrue(all(queue.read(Path(r["directory"]) / "status.json")["state"] == "completed" for r in runs))

    def test_failed_batch_does_not_skip_other_finished_worker(self):
        runner = queue.Queue(self.manifest)
        runs = self.manifest["runs"][:2]
        self.manifest["settings"] = {**queue.SETTINGS, "worker_stagger_seconds": 0}
        self.complete(runs[1])
        # Ensure the second run is launched, rather than already-complete skipping.
        (Path(runs[1]["directory"]) / "done.json").unlink()
        def launch(command, **kwargs):
            target = command[command.index("--out") + 1]
            if target == runs[1]["directory"]:
                self.complete(runs[1])
            return Mock(pid=424242, poll=Mock(return_value=1 if target == runs[0]["directory"] else 0))
        with patch.object(queue.subprocess, "Popen", side_effect=launch), self.assertRaisesRegex(ValueError, "Batch failed"):
            runner.batch("qwen-llama", runs)
        self.assertEqual(queue.read(Path(runs[0]["directory"]) / "status.json")["state"], "failed")
        self.assertEqual(queue.read(Path(runs[1]["directory"]) / "status.json")["state"], "completed")

    def test_dead_tunnel_stops_waiting_for_optimizers(self):
        runner = queue.Queue(self.manifest)
        runner.tunnel = Mock(poll=Mock(return_value=255))
        worker = Mock(pid=424242, poll=Mock(return_value=None))
        run = self.manifest["runs"][0]
        with patch.object(queue.subprocess, "Popen", return_value=worker), self.assertRaisesRegex(
                ValueError, "Bridge tunnel exited"):
            runner.batch("qwen-llama", [run])
        # The enclosing run() finally block owns termination/cleanup. Keep the
        # worker recorded so that it can be stopped before cancelling serving.
        self.assertEqual(runner.workers, [worker])


if __name__ == "__main__":
    unittest.main()
