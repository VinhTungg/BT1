"""Evaluate fixed baseline configurations after sealing validation selection."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import train_baselines as base


def evaluate_run(dataset_path: Path, model_manifest_path: Path, report_dir: Path) -> dict:
    if report_dir.exists():
        raise ValueError('Evaluation directory already exists; retain the sealed evaluation.')
    dataset = json.loads(dataset_path.read_text(encoding='utf-8'))
    manifest = json.loads(model_manifest_path.read_text(encoding='utf-8'))
    paths = {name: ROOT / dataset['splits'][name]['path'] for name in ('train', 'val', 'test')}
    for name, path in paths.items():
        if base.file_hash(path) != dataset['splits'][name]['sha256']:
            raise ValueError(f'Dataset fingerprint differs: {name}')
    for name, model_key in (('train', 'train'), ('val', 'validation')):
        if manifest[model_key]['sha256'] != dataset['splits'][name]['sha256']:
            raise ValueError(f'Model was fitted/selected on different {name} data')
    train, val = base.load_split(paths['train']), base.load_split(paths['val'])
    base.validate_separation(train, val, 'text_tokenized')
    if {m['model'] for m in manifest['models']} != set(base.FAMILIES) | {'dummy'}:
        raise ValueError('All four required baselines and Dummy are required')
    loaded = {}
    for metadata in manifest['models']:
        artifact = Path(metadata['artifact'])
        if base.file_hash(artifact) != metadata['sha256']:
            raise ValueError(f'Artifact fingerprint differs: {metadata["model"]}')
        model = joblib.load(artifact)
        metrics = base.evaluate(val['label'], model.predict(val['text_tokenized'].tolist()))
        if metrics != metadata['validation_metrics']:
            raise ValueError(f'Validation results differ: {metadata["model"]}')
        loaded[metadata['model']] = model
    selected = max((m for m in manifest['models'] if m['model'] in base.FAMILIES),
                   key=lambda m: base.selection_key(m['validation_metrics']))
    if selected['model'] != manifest['best_baseline_on_validation']:
        raise ValueError('Manifest winner differs from the validation selection rule')
    selection = {
        'sealed_at_utc': datetime.now(timezone.utc).isoformat(),
        'dataset_manifest': str(dataset_path.resolve()),
        'dataset_manifest_sha256': base.file_hash(dataset_path),
        'model_manifest': str(model_manifest_path.resolve()),
        'model_manifest_sha256': base.file_hash(model_manifest_path),
        'selection_rule': manifest['selection_rule'], 'selected_model': selected,
        'test_sha256': dataset['splits']['test']['sha256'],
        'all_fixed_models': manifest['models'],
        'evaluation_policy': dataset.get('evaluation_policy', ''),
        'label_policy': dataset.get('label_policy', ''),
    }
    report_dir.mkdir(parents=True, exist_ok=False)
    base.write_json(report_dir / 'selection_before_test.json', selection)
    test = base.load_split(paths['test'])
    if len(test) != dataset['splits']['test']['samples']:
        raise ValueError('Test row count differs from the dataset manifest')
    for frame in (train, val):
        base.validate_separation(frame, test, 'text_tokenized')
        if set(frame['post_id'].str.rsplit(':', n=1).str[-1]) & set(test['post_id'].str.rsplit(':', n=1).str[-1]):
            raise ValueError('Numeric source-post overlap with test')
    records, predictions = [], test.copy()
    for metadata in manifest['models']:
        name = metadata['model']
        predicted = loaded[name].predict(test['text_tokenized'].tolist())
        predictions[f'prediction_{name}'] = predicted
        records.append({'model': name, 'artifact': metadata['artifact'], 'sha256': metadata['sha256'],
                        'validation': metadata['validation_metrics'],
                        'test': base.evaluate(test['label'], predicted)})
    result = {
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'selection_sha256': base.file_hash(report_dir / 'selection_before_test.json'),
        'selected_model': selected['model'], 'selection_basis': 'validation only, excluding Dummy',
        'train_rows': len(train), 'validation_rows': len(val), 'test_rows': len(test),
        'test_posts': int(test['post_id'].nunique()),
        'selected_test_metrics': next(m['test'] for m in records if m['model'] == selected['model']),
        'models': records, 'evaluation_policy': selection['evaluation_policy'],
        'label_policy': selection['label_policy'],
    }
    base.write_csv(report_dir / 'test_predictions.csv', predictions)
    base.write_json(report_dir / 'results.json', result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-manifest', required=True, type=Path)
    parser.add_argument('--model-manifest', required=True, type=Path)
    parser.add_argument('--report-dir', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = evaluate_run(args.dataset_manifest, args.model_manifest, args.report_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f'Baseline evaluation failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
