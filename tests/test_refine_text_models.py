import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC

from scripts import refine_text_models as refine
from scripts import train_baselines as base
from src.decision_threshold import DecisionThresholdClassifier, select_threshold
from src.text_features import build_features


class RefinementTests(unittest.TestCase):
    def test_eight_variants_preserve_negation_and_units(self):
        specs = list(refine.controlled_specs())
        self.assertEqual(len(specs), 8)
        frame = pd.DataFrame({'text_clean':['và không 100k', 'và không 100k'],
                              'text_tokenized':['và không 100k', 'và không 100k']})
        features = build_features('word_char', word_stop_words=refine.STOP_WORDS).fit(frame)
        words = features.named_transformers_['word'].vocabulary_
        self.assertNotIn('và', words)
        self.assertIn('không', words)
        self.assertIn('100k', words)

    def test_threshold_is_data_dependent_and_wrapper_uses_it(self):
        best, rows = select_threshold([0,0,1,1], [-2.,-1.,-.2,-.1])
        self.assertLess(best['threshold'], 0)
        self.assertEqual(best['macro_f1'], 1)
        self.assertTrue(any(r['threshold'] == 0 for r in rows))
        with self.assertRaises(ValueError):
            select_threshold([0,1], [np.nan, 1])
        model = Pipeline([('vectorizer',CountVectorizer()),('classifier',LinearSVC())]).fit(['good good','bad bad'],[0,1])
        wrapper = DecisionThresholdClassifier(model, threshold=100)
        self.assertEqual(wrapper.predict(['bad']).tolist(), [0])

    def test_grouped_run_never_opens_test_or_changes_source_rows(self):
        with tempfile.TemporaryDirectory(dir=base.ROOT) as directory:
            root = Path(directory)
            paths = {}
            for split, count in [('train',12),('val',4)]:
                frame = pd.DataFrame([dict(id=f'{split}{i}',post_id=f'{split}:{i//2}',label=i%2,
                                          text=('đồ ngốc ko tốt' if i%2 else 'học phí bình thường')+f' {split}only{i}',
                                          text_tokenized=f'{split}only{i}',text_clean=f'{split}only{i}') for i in range(count)])
                paths[split] = root/f'{split}.csv';frame.to_csv(paths[split], index=False)
            defaults = dict(train=paths['train'],val=paths['val'])
            source = root/'models/resplit_v1/steps23_v1/manifest.json';source.parent.mkdir(parents=True)
            source.write_text(json.dumps({'train':{'sha256':base.file_hash(paths['train'])},
                                           'validation':{'sha256':base.file_hash(paths['val'])}}), encoding='utf-8')
            before = {p:base.file_hash(p) for p in paths.values()}
            spec = next(refine.controlled_specs())
            with patch.object(refine,'ROOT',root), patch.object(base,'training_defaults',return_value=defaults), \
                 patch.object(base.pd,'read_csv',wraps=pd.read_csv) as read:
                summary = refine.run(root/'models/new',root/'reports/new',specs=[spec])
            self.assertEqual([c.args[0] for c in read.call_args_list], [paths['train'],paths['val']])
            self.assertFalse(summary['test_evaluated'])
            self.assertEqual(len(summary['models']),2)
            self.assertTrue(all(m['reload_verified'] for m in summary['models']))
            self.assertTrue(all(base.file_hash(p) == digest for p,digest in before.items()))
            assignments = pd.read_csv(root/'reports/new/train_fold_assignments.csv')
            self.assertEqual(assignments.groupby('post_id').fold.nunique().max(),1)


if __name__ == '__main__':
    unittest.main()
