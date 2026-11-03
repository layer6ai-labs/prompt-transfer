"""Overview integrity: pending test results and same-snapshot cost views."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import migration_overview as overview
from test_migration_plots import fixture


class OverviewIntegrityTest(unittest.TestCase):
    def test_pending_optimizer_cost_is_unknown_then_training_only_is_labeled(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            report = fixture(complete=False)
            arm = report['routes'][0]['arms'][3]
            arm['optimization_dir'] = str(directory)
            row = overview.overview_records(report)[3]
            self.assertIsNone(row['migration_flops'])
            self.assertIsNone(row['end_to_end_flops'])
            self.assertEqual(row['optimization_status'], 'pending')
            (directory / 'done.json').write_text('{}')
            row = overview.overview_records(report)[3]
            self.assertEqual(row['migration_pflop'], 2)
            self.assertEqual(row['end_to_end_pflop'], 3)
            self.assertTrue(row['training_complete_evaluation_pending'])
            self.assertEqual(row['optimization_display_status'], 'training complete / evaluation pending')
            self.assertIsNone(row['score_pct'])
            self.assertIsNone(row['inference_tflop_per_item'])
            self.assertIsNotNone(row['optimization_done_sha256'])

    def test_both_views_use_one_fresh_report_and_identical_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / 'manifest.json'
            manifest.write_text('{}')
            report = fixture(complete=False)
            seen = []
            def render(actual_report, rows, output, scope, created):
                seen.append((actual_report, rows, created, scope))
                return {'scope': scope, 'png': str(output / f'overview_{scope}.png'),
                        'pdf': str(output / f'overview_{scope}.pdf'), 'notes': []}
            with patch.object(overview, 'build_report', return_value=report) as build:
                with patch.object(overview, '_render_sheet', side_effect=render):
                    metadata = overview.generate_overview(manifest, root / 'visualization')
            self.assertEqual(build.call_count, 1)
            self.assertEqual([value[3] for value in seen], ['end_to_end', 'migration'])
            self.assertIs(seen[0][0], seen[1][0])
            self.assertIs(seen[0][1], seen[1][1])
            self.assertEqual(seen[0][2], seen[1][2])
            self.assertEqual(metadata['complete_method_cells'], 0)
            snapshot = json.loads((root / 'visualization/plot_data.json').read_text())
            self.assertEqual(snapshot['cost_scopes'], ['end_to_end', 'migration'])
            self.assertTrue(all(row['score_pct'] is None for row in snapshot['records']))


if __name__ == '__main__':
    unittest.main()
