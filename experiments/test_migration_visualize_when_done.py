"""Verify that visualization cannot start from partial or stale results."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import migration_visualize_when_done as observer
import migration_queue


class VisualizationObserverTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {"queue_id": "campaign"}
        self.state = {"queue_id": "campaign", "state": "completed",
                      "stages": {s: "completed" for s in migration_queue.STAGES}, "pairs": {}}

    def test_stale_state_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "different frozen manifest"):
            observer.campaign_readiness(self.manifest, {**self.state, "queue_id": "other"})

    def test_running_or_failed_campaign_stays_pending(self):
        for value in ("running", "failed"):
            state = {**self.state, "state": value}
            with patch.object(migration_queue, "is_complete") as complete:
                result = observer.campaign_readiness(self.manifest, state)
            self.assertEqual(result["state"], "waiting_for_campaign")
            complete.assert_not_called()

    def test_partial_stages_and_missing_outputs_cannot_generate(self):
        with self.assertRaisesRegex(ValueError, "all three stages"):
            observer.campaign_readiness(self.manifest, {**self.state, "stages": {"eval123": "completed"}})
        with patch.object(migration_queue, "verify_manifest"), patch.object(migration_queue, "jobs_for", return_value=[({}, {})]), patch.object(migration_queue, "is_complete", return_value=False):
            with self.assertRaisesRegex(ValueError, "Missing validated"):
                observer.campaign_readiness(self.manifest, self.state)

    def test_cleanup_is_required_before_plots(self):
        self.state['pairs'] = {'eval4': {'serve_job_id': 99, 'serve_cancelled': False}}
        self.assertEqual(observer.campaign_readiness(self.manifest, self.state)['state'], 'waiting_for_cleanup')

    def test_completion_marker_requires_intact_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / 'figure.png'
            artifact.write_bytes(b'valid artifact')
            status = {'state': 'completed', 'queue_id': 'campaign', 'artifact_sha256': {str(artifact): observer.digest(artifact)}}
            self.assertTrue(observer._existing_artifacts_valid(status, 'campaign'))
            artifact.write_bytes(b'changed')
            self.assertFalse(observer._existing_artifacts_valid(status, 'campaign'))


if __name__ == '__main__':
    unittest.main()
