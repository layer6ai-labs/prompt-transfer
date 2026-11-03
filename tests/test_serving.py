"""Serving preflight checks, using tiny checkpoints and no GPU or network."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "serving" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validator = load_module("validate_model")
smoke = load_module("smoke")


def checkpoint(directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text('{}')
    header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    header += b" " * (-len(header) % 8)
    (directory / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)
    return directory


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = checkpoint(Path(self.tmp.name))

    def test_valid_single_and_indexed_checkpoints(self):
        self.assertEqual(validator.validate_model(self.directory)[0], 1)
        (self.directory / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"weight": "model.safetensors"}}))
        self.assertEqual(validator.validate_model(self.directory)[0], 1)

    def test_nonempty_truncated_shard_is_rejected(self):
        shard = self.directory / "model.safetensors"
        shard.write_bytes(shard.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "tensor payload"):
            validator.validate_model(self.directory)

    def test_truncated_header_is_rejected(self):
        (self.directory / "model.safetensors").write_bytes(struct.pack("<Q", 1000) + b"{}")
        with self.assertRaisesRegex(ValueError, "truncated safetensors header"):
            validator.validate_model(self.directory)

    def test_missing_indexed_shard_is_rejected(self):
        (self.directory / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"weight": "missing.safetensors"}}))
        with self.assertRaisesRegex(ValueError, "incomplete shard set"):
            validator.validate_model(self.directory)

    def test_malformed_index_returns_failure(self):
        (self.directory / "model.safetensors.index.json").write_text('[]')
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(validator.main([str(self.directory)]), 1)


class LlamaFetchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.ptx = self.base / "ptx"
        self.env = {**os.environ, "PTX_HOME": str(self.ptx)}
        self.vendor = self.ptx / "vendor" / "llama-3.1-official"
        self.vendor.mkdir(parents=True)
        for name in ("config.json", "generation_config.json", "tokenizer.json",
                     "tokenizer_config.json", "special_tokens_map.json"):
            (self.vendor / name).write_text('{}')
        (self.vendor / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"weight": "model.safetensors"}}))
        bin_dir = self.ptx / ".venv-sglang" / "bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "python").symlink_to(sys.executable)
        hf = bin_dir / "hf"
        hf.write_text(
            f"#!{sys.executable}\n"
            "import os, shutil, sys\n"
            "from pathlib import Path\n"
            "Path(os.environ['FETCH_TEST_CALLED']).touch()\n"
            "shutil.copytree(os.environ['FETCH_TEST_SOURCE'], sys.argv[-1], dirs_exist_ok=True)\n")
        hf.chmod(0o755)
        self.source = checkpoint(self.base / "download")
        self.called = self.base / "download-called"
        self.env.update(FETCH_TEST_SOURCE=str(self.source), FETCH_TEST_CALLED=str(self.called))
        self.small = "Llama-3.1-8B-Instruct"
        self.large = "Llama-3.1-70B-Instruct-FP8"
        self.seed_cached(self.small)

    def revision(self, name):
        result = subprocess.run(
            ["bash", "-c", 'source "$1"; printf "%s" "${MODEL_REV[$2]}"',
             "bash", str(ROOT / "serving" / "models.sh"), name],
            env=self.env, capture_output=True, text=True, check=True)
        return result.stdout

    def seed_cached(self, name):
        directory = checkpoint(self.ptx / "models" / name)
        for path in self.vendor.iterdir():
            shutil.copyfile(path, directory / path.name)
        (directory / "REVISION").write_text(self.revision(name))
        return directory

    def run_fetch(self):
        return subprocess.run(["bash", str(ROOT / "serving" / "fetch_llama31.sh")],
                              env=self.env, capture_output=True, text=True)

    def test_successful_download_is_marked_after_validation(self):
        result = self.run_fetch()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(self.called.exists())
        directory = self.ptx / "models" / self.large
        self.assertEqual((directory / "REVISION").read_text().strip(), self.revision(self.large))
        validator.validate_model(directory)

    def test_generic_setup_rejects_llama_before_provisioning(self):
        shutil.rmtree(self.ptx / ".venv-sglang")
        result = subprocess.run(["bash", str(ROOT / "serving" / "setup.sh"),
                                 "--layout", "llama31"],
                                env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("serving/fetch_llama31.sh", result.stderr)
        self.assertFalse((self.ptx / ".venv-sglang").exists())

    def test_bad_cached_large_checkpoint_is_refetched_and_not_marked(self):
        directory = self.seed_cached(self.large)
        for path in (directory / "model.safetensors", self.source / "model.safetensors"):
            path.write_bytes(path.read_bytes()[:-1])
        result = self.run_fetch()
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(self.called.exists())
        self.assertFalse((directory / "REVISION").exists())

    def test_bad_cached_small_checkpoint_cannot_keep_revision_marker(self):
        directory = self.ptx / "models" / self.small
        shard = directory / "model.safetensors"
        shard.write_bytes(shard.read_bytes()[:-1])
        result = self.run_fetch()
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(self.called.exists())
        self.assertFalse((directory / "REVISION").exists())


class SmokeTests(unittest.TestCase):
    def test_harmony_requests_low_effort_instead_of_disabling_thinking(self):
        response = {"choices": [{"message": {"content": "391", "reasoning_content": "math"}}]}
        with patch.object(smoke.urllib.request, "urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value = io.BytesIO(json.dumps(response).encode())
            result = smoke.chat("http://example.test/v1", "gpt-oss-20b", "question", 32)
        body = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(body["reasoning_effort"], "low")
        self.assertEqual(body["max_tokens"], 2048)
        self.assertNotIn("chat_template_kwargs", body)
        self.assertEqual(result["reasoning"], "math")

    def test_other_model_completion_budgets_are_unchanged(self):
        response = {"choices": [{"message": {"content": "391"}}]}
        with patch.object(smoke.urllib.request, "urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value = io.BytesIO(json.dumps(response).encode())
            smoke.chat("http://example.test/v1", "Qwen3-8B", "question", 32)
        body = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(body["max_tokens"], 32)
        self.assertNotIn("reasoning_effort", body)
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})

    def test_missing_requested_model_is_rejected_before_inference(self):
        with patch.object(smoke, "chat") as chat, \
                patch.object(smoke.urllib.request, "urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value = io.BytesIO(
                json.dumps({"data": [{"id": "another-model"}]}).encode())
            with self.assertRaisesRegex(ValueError, "absent from /v1/models"):
                smoke.check("Qwen3-8B", "http://example.test/v1")
            chat.assert_not_called()

    def test_harmony_reasoning_is_separate_and_other_models_must_disable_it(self):
        reply = {"text": "391", "reasoning": "math", "prompt_tokens": 3000,
                 "completion_tokens": 1, "seconds": 1}
        for model, expected in (("gpt-oss-20b", True), ("Qwen3-8B", False)):
            with self.subTest(model=model), patch.object(smoke, "chat", return_value=reply), \
                    patch.object(smoke.urllib.request, "urlopen") as urlopen:
                urlopen.return_value.__enter__.return_value = io.BytesIO(
                    json.dumps({"data": [{"id": model}]}).encode())
                result = smoke.check(model, "http://example.test/v1")
            self.assertEqual(result["reasoning_ok"], expected)
            self.assertEqual(smoke.checks_passed(result), expected)

    def run_main(self, result, endpoints=None):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "endpoints.json").write_text(json.dumps(
                {"test-model": "http://example.test/v1"} if endpoints is None else endpoints))
            with patch.object(smoke, "PTX_HOME", str(directory)), \
                    patch.object(smoke, "check", return_value=result), \
                    contextlib.redirect_stdout(io.StringIO()):
                code = smoke.main(["--out", str(directory / "result.json")])
            return code, json.loads((directory / "result.json").read_text())

    def passing_result(self):
        return {"model": "test-model", "answer_ok": True, "answer_text": "391",
                "reasoning_ok": True, "repeat_identical": True, "long_identical": True,
                "long_prompt_tokens": 3000, "stress_identical": 48, "stress_total": 48,
                "throughput_tok_s": 10, "cot_tail": "Answer: 72"}

    def test_failed_determinism_exits_nonzero_and_preserves_report(self):
        result = {**self.passing_result(), "stress_identical": 47}
        code, saved = self.run_main(result)
        self.assertEqual(code, 1)
        self.assertEqual(saved[0]["stress_identical"], 47)

    def test_request_error_exits_nonzero(self):
        self.assertEqual(self.run_main({"model": "test-model", "error": "timeout"})[0], 1)

    def test_all_checks_pass_exits_zero(self):
        self.assertEqual(self.run_main(self.passing_result())[0], 0)

    def test_empty_layout_is_not_a_successful_smoke(self):
        self.assertEqual(self.run_main({}, endpoints={})[0], 1)


if __name__ == "__main__":
    unittest.main()
