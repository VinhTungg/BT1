"""Seal validation selection, then compare saved old/new models on frozen test once."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import train_baselines as base

METRICS = ('accuracy', 'precision_toxic', 'recall_toxic', 'f1_toxic', 'macro_f1')


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def inputs(model, frame):
    if 'features' in model.named_steps:
        return frame[['text_clean', 'text_tokenized']]
    return frame['text_tokenized'].tolist()


def load_artifact(metadata: dict):
    path = Path(metadata.get('artifact') or metadata['path'])
    if fingerprint(path) != metadata['sha256']:
        raise ValueError(f'Artifact fingerprint differs: {path}')
    return joblib.load(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT/'data/expanded/round_2')
    parser.add_argument('--reference', type=Path, default=ROOT/'reports/data_expansion/reference_run.json')
    parser.add_argument('--baselines', type=Path, default=ROOT/'models/data_expansion/round_2/baselines/manifest.json')
    parser.add_argument('--improvements', type=Path, default=ROOT/'models/data_expansion/round_2/improvements/manifest.json')
    parser.add_argument('--report-dir', type=Path, default=ROOT/'reports/data_expansion/round_2/final_evaluation')
    args = parser.parse_args(argv)
    try:
        if args.report_dir.exists():
            raise ValueError('Final evaluation directory already exists; do not repeat test-driven selection.')
        frozen = read_json(ROOT/'data/manifest.json')
        dataset = read_json(args.dataset/'manifest.json')
        reference = read_json(args.reference)
        manifests = [read_json(args.baselines), read_json(args.improvements)]
        train_path = args.dataset/'train.csv'
        if (dataset['ready_new_rows'] < 6000 or dataset['audit_only']
                or fingerprint(train_path) != dataset['train_sha256']):
            raise ValueError('Expanded training dataset is incomplete or changed')
        for name in ('train', 'val', 'test'):
            if fingerprint(ROOT/frozen['splits'][name]['path']) != frozen['splits'][name]['sha256']:
                raise ValueError(f'Frozen split changed: {name}')
        old_train = base.load_split(ROOT/frozen['splits']['train']['path'])
        train = base.load_split(train_path)
        val = base.load_split(ROOT/frozen['splits']['val']['path'])
        if (len(train) - len(old_train) != dataset['ready_new_rows']
                or not train.iloc[:len(old_train)].reset_index(drop=True).equals(old_train)):
            raise ValueError('Expanded train does not preserve the original training prefix')
        base.validate_separation(train, val, 'text_tokenized')
        if reference['val_sha256'] != frozen['splits']['val']['sha256']:
            raise ValueError('Reference used a different validation set')
        old_meta = next(m for m in reference['models'] if m['model'] == reference['best_on_validation'])
        old_model = load_artifact(old_meta)
        old_val = base.evaluate(val['label'], old_model.predict(inputs(old_model, val)))
        if old_val != old_meta['validation_metrics']:
            raise ValueError('Old model does not reproduce its validation scores')
        candidates = []
        for manifest_path, manifest in zip((args.baselines, args.improvements), manifests):
            if (manifest['train']['sha256'] != dataset['train_sha256']
                    or manifest['validation']['sha256'] != frozen['splits']['val']['sha256']):
                raise ValueError('Candidate training/validation fingerprint differs')
            for metadata in manifest['models']:
                model = load_artifact(metadata)
                metrics = base.evaluate(val['label'], model.predict(inputs(model, val)))
                if metrics != metadata['validation_metrics']:
                    raise ValueError(f'Candidate validation scores differ: {metadata["model"]}')
                candidates.append((metadata, model, metrics, manifest_path))
        selected, new_model, new_val, selected_manifest = max(candidates, key=lambda item: base.selection_key(item[2]))
        selection = {
            'sealed_at_utc': datetime.now(timezone.utc).isoformat(),
            'selection_rule': manifests[0]['selection_rule'],
            'dataset_manifest_sha256': fingerprint(args.dataset/'manifest.json'),
            'reference_manifest_sha256': fingerprint(args.reference),
            'candidate_manifest_sha256': {str(p): fingerprint(p) for p in (args.baselines, args.improvements)},
            'reference_model': old_meta, 'selected_model': selected,
            'selected_manifest': str(selected_manifest),
            'validation': {'reference': old_val, 'selected': new_val},
            'all_candidate_validation_scores': [{'model': m['model'], **metrics} for m, _, metrics, _ in candidates],
            'test_fingerprint': frozen['splits']['test']['sha256'],
            'test_predictions_generated': False,
        }
        # Reserve this evaluation before reading predictions; never switch models after test.
        args.report_dir.mkdir(parents=True, exist_ok=False)
        (args.report_dir/'selection_before_test.json').write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding='utf-8')
        test = base.load_split(ROOT/frozen['splits']['test']['path'])
        base.validate_separation(train, test, 'text_tokenized')
        base.validate_separation(val, test, 'text_tokenized')
        old_predictions = old_model.predict(inputs(old_model, test))
        new_predictions = new_model.predict(inputs(new_model, test))
        old_test = base.evaluate(test['label'], old_predictions)
        new_test = base.evaluate(test['label'], new_predictions)
        results = {
            'completed_at_utc': datetime.now(timezone.utc).isoformat(),
            'selection_sha256': fingerprint(args.report_dir/'selection_before_test.json'),
            'reference_model': old_meta['model'], 'selected_model': selected['model'],
            'train_rows': len(train), 'new_train_rows': dataset['ready_new_rows'],
            'validation_rows': len(val), 'test_rows': len(test),
            'validation': selection['validation'], 'test': {'reference': old_test, 'selected': new_test},
            'validation_delta': {k: new_val[k] - old_val[k] for k in METRICS},
            'test_delta': {k: new_test[k] - old_test[k] for k in METRICS},
            'all_five_metrics_improved_on_validation': all(new_val[k] > old_val[k] for k in METRICS),
            'all_five_metrics_improved_on_test': all(new_test[k] > old_test[k] for k in METRICS),
            'limitations': ['New labels are AI-reviewed silver labels, not independent human gold.',
                            'Validation and test are small, post-disjoint frozen sets with inherited labels.',
                            'Hyperparameters selected on validation; test was used after sealing selection.'],
        }
        predictions = test.copy()
        predictions['reference_prediction'] = old_predictions
        predictions['selected_prediction'] = new_predictions
        base.write_csv(args.report_dir/'test_predictions.csv', predictions)
        (args.report_dir/'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f'Final expansion evaluation failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
