"""Default-path selection uses the active dataset, without reading its test CSV."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import train_baselines as base


class ActiveDatasetTests(unittest.TestCase):
    def test_no_pointer_keeps_legacy_defaults(self):
        with tempfile.TemporaryDirectory(dir=base.ROOT) as directory:
            root = Path(directory)
            with patch.object(base, 'ROOT', root):
                self.assertEqual(base.training_defaults()['train'], root/'data/splits/train.csv')

    def test_active_paths_verified_without_opening_test(self):
        with tempfile.TemporaryDirectory(dir=base.ROOT) as directory:
            root = Path(directory)
            (root/'data').mkdir()
            splits = {}
            for name in ('train', 'val'):
                path = root/f'{name}.csv'
                path.write_text(name, encoding='utf-8')
                splits[name] = {'path': path.name, 'sha256': base.file_hash(path)}
            splits['test'] = {'path': 'nonexistent_test.csv', 'sha256': 'not-read'}
            manifest = root/'dataset.json'
            manifest.write_text(json.dumps({'splits': splits}), encoding='utf-8')
            active = {'manifest': manifest.name, 'manifest_sha256': base.file_hash(manifest),
                      'outputs': {name: name for name in ('baseline_models', 'baseline_reports', 'improved_models', 'improved_reports')}}
            (root/'data/current_dataset.json').write_text(json.dumps(active), encoding='utf-8')
            with patch.object(base, 'ROOT', root):
                self.assertEqual(base.training_defaults()['val'], root/'val.csv')
                (root/'train.csv').write_text('changed', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'Active train dataset changed'):
                    base.training_defaults()


if __name__ == '__main__':
    unittest.main()
