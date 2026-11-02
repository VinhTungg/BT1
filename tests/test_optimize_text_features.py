"""The optimization run must isolate holdouts and preserve source datasets."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts import optimize_text_features as optimize
from scripts import train_baselines as base


class OptimizeFeatureTests(unittest.TestCase):
    def test_fixed_normalization_collisions_are_reported_without_dropping_rows(self):
        train = pd.DataFrame(dict(text_clean=['không biết'], text_tokenized=['không biết']))
        val = train.copy()
        self.assertEqual(optimize.normalization_overlap(train, val), dict(text_clean=1, text_tokenized=1))
        self.assertEqual(len(val), 1)

    def test_fixed_grid_is_bounded_and_paired(self):
        configs = list(optimize.feature_configs())
        self.assertEqual(sum(len(list(optimize.classifier_configs(c['kind']))) for c in configs) * 2, 104)
        word = {kind: [(c['ngram_range'], c['min_df']) for c in configs if c['kind'] == kind]
                for kind in ('bow', 'tfidf')}
        self.assertEqual(word['bow'], word['tfidf'])

    def test_end_to_end_uses_only_train_validation_and_keeps_source_files(self):
        with tempfile.TemporaryDirectory(dir=base.ROOT) as directory:
            root = Path(directory)
            paths = {}
            for name, count in [('train', 12), ('val', 4)]:
                frame = pd.DataFrame([
                    dict(id=f'{name}{i}', post_id=f'{name}_post{i//2}', label=i % 2,
                         text=('đồ ngốc ko tốt' if i % 2 else 'học phí bình thường') + f' {name}only{i}',
                         text_clean=f'{name}clean{i}',
                         text_tokenized=('đồ_ngốc' if i % 2 else 'học_phí') + f' {name}only{i}')
                    for i in range(count)
                ])
                paths[name] = root / f'{name}.csv'
                frame.to_csv(paths[name], index=False)
            baseline_dir = root / 'baselines'
            base.run_training(paths['train'], paths['val'], baseline_dir, root / 'baseline_reports', quick=True)
            before = {p: base.file_hash(p) for p in paths.values()}
            defaults = dict(train=paths['train'], val=paths['val'], baseline_models=baseline_dir)
            with patch.object(base, 'training_defaults', return_value=defaults), \
                 patch.object(base.pd, 'read_csv', wraps=pd.read_csv) as read:
                summary = optimize.run((root / 'optimized_models').relative_to(base.ROOT),
                                       (root / 'optimized_reports').relative_to(base.ROOT), quick=True, promote=False)
            self.assertEqual([call.args[0] for call in read.call_args_list], [paths['train'], paths['val']])
            self.assertTrue(all(base.file_hash(p) == digest for p, digest in before.items()))
            self.assertEqual(summary['trials'], 12)
            self.assertEqual(len(summary['models']), 12)
            self.assertFalse(summary['test_evaluated'])
            self.assertFalse(summary['promoted'])
            self.assertTrue(all(m['reload_verified'] for m in summary['models']))
            saved = json.loads((root / 'optimized_models/manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['train']['sha256'], before[paths['train']])
            with self.assertRaisesRegex(ValueError, 'already exists'):
                optimize.run(root / 'optimized_models', root / 'optimized_reports', quick=True)


if __name__ == '__main__':
    unittest.main()
