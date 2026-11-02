"""Local synthetic fixtures; never read or score the project's real test set."""
import json
import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from scripts import train_baselines as base
from scripts.evaluate_baselines import evaluate_run


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=base.ROOT)
        self.root = Path(self.temp.name)
        self.dataset = self.root/'dataset.json'
        self.models = self.root/'models.json'
        self.output = self.root/'evaluation'
        splits = {}
        for i, name in enumerate(('train', 'val', 'test'), 1):
            labels = [0, 0, 1] if name != 'test' else [0, 1, 1]
            frame = pd.DataFrame([{'id': f'{name}{j}', 'post_id': f'fixture:{i}',
                                   'text_clean': f'{name}sample{j}', 'text_tokenized': f'{name}sample{j}',
                                   'label': label} for j, label in enumerate(labels)])
            path = self.root/f'{name}.csv'
            frame.to_csv(path, index=False)
            splits[name] = {'path': str(path), 'sha256': base.file_hash(path), 'samples': len(frame)}
        self.dataset.write_text(json.dumps({'splits': splits}), encoding='utf-8')
        records = []
        # Dummy pipelines isolate selection logic: class 0 wins validation, class 1 wins test.
        for name in (*base.FAMILIES, 'dummy'):
            constant = 0 if name in ('bow_nb', 'dummy') else 1
            model = Pipeline([('classifier', DummyClassifier(strategy='constant', constant=constant))])
            model.fit(['a', 'b', 'c'], [0, 0, 1])
            path = self.root/f'{name}.joblib'
            joblib.dump(model, path)
            records.append({'model': name, 'artifact': str(path), 'sha256': base.file_hash(path),
                            'validation_metrics': base.evaluate([0, 0, 1], model.predict(['a', 'b', 'c']))})
        self.models.write_text(json.dumps({'train': splits['train'], 'validation': splits['val'],
                                          'models': records, 'best_baseline_on_validation': 'bow_nb',
                                          'selection_rule': 'macro_f1, f1_toxic, recall_toxic'}), encoding='utf-8')

    def tearDown(self):
        self.temp.cleanup()

    def test_selection_remains_validation_winner_even_when_another_model_wins_test(self):
        result = evaluate_run(self.dataset, self.models, self.output)
        self.assertEqual(result['selected_model'], 'bow_nb')
        others = [r for r in result['models'] if r['model'] == 'tfidf_svm']
        self.assertGreater(others[0]['test']['macro_f1'], result['selected_test_metrics']['macro_f1'])
        with self.assertRaisesRegex(ValueError, 'already exists'):
            evaluate_run(self.dataset, self.models, self.output)

    def test_changed_dataset_fails_before_sealing(self):
        (self.root/'test.csv').write_text('changed', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Dataset fingerprint differs'):
            evaluate_run(self.dataset, self.models, self.output)
        self.assertFalse(self.output.exists())

    def test_changed_artifact_fails_before_sealing(self):
        (self.root/'bow_nb.joblib').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Artifact fingerprint differs'):
            evaluate_run(self.dataset, self.models, self.output)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
