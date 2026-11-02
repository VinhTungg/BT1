"""Three grouped folds, eight controlled variants, and train-only thresholds."""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import train_baselines as base
from src.decision_threshold import DecisionThresholdClassifier, select_threshold
from src.text_features import CommentPreprocessor, build_features

LOG = logging.getLogger('refine_text_models')
STOP_WORDS = ['và', 'là', 'của', 'ở', 'tại', 'trong', 'trên', 'dưới', 'theo']
REFERENCES = ['slang_word_char_svm', 'preserve_bow_svm', 'slang_tfidf_svm']


def controlled_specs():
    for profile, maximum, remove_stops in itertools.product(('preserve', 'slang'), (2, 3), (False, True)):
        yield dict(name=f'{profile}_ng{maximum}_' + ('stop' if remove_stops else 'keep'),
                   experiment='controlled', profile=profile,
                   feature_params=dict(kind='word_char', ngram_range=(1, maximum), min_df=2,
                                       char_weight=1.0, word_stop_words=STOP_WORDS if remove_stops else None),
                   classifier_params=dict(C=0.1, class_weight='balanced'))


def group_folds(train, seed=42):
    groups = train.post_id.str.rsplit(':', n=1).str[-1]
    if groups.nunique() < 3:
        raise ValueError('At least three source posts required')
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    result = list(splitter.split(train, train.label, groups=groups))
    counts = np.zeros(len(train), dtype=int)
    for fit, held in result:
        if set(groups.iloc[fit]) & set(groups.iloc[held]):
            raise RuntimeError('Source post leakage in folds')
        if set(train.label.iloc[fit]) != {0, 1} or set(train.label.iloc[held]) != {0, 1}:
            raise ValueError('Both classes required in every fold')
        counts[held] += 1
    if not np.all(counts == 1):
        raise RuntimeError('Every training row must be held out exactly once')
    return result


def _features(params):
    params = dict(params)
    for key in ('ngram_range', 'char_range'):
        if key in params:
            params[key] = tuple(params[key])
    return build_features(**params)


def prediction_table(frame, scores, threshold):
    predicted = (np.asarray(scores) >= threshold).astype(int)
    result = frame[['id', 'post_id', 'text', 'label']].copy()
    result['score'] = scores
    result['prediction'] = predicted
    result['outcome'] = [('TP' if a else 'TN') if a == b else ('FN' if a else 'FP')
                         for a, b in zip(frame.label, predicted)]
    result['has_negation'] = frame.text.str.contains(r'\b(?:không|ko|đừng|chẳng|chưa|khum|hok)\b', case=False)
    result['has_mention'] = frame.text.str.contains('[MENTION]', regex=False)
    result['short_comment'] = frame.text.str.split().str.len().le(5)
    return result


def run(model_dir, report_dir, *, specs=None, folds=None):
    model_dir, report_dir = Path(model_dir).resolve(), Path(report_dir).resolve()
    if any(not p.is_relative_to(ROOT) for p in (model_dir, report_dir)):
        raise ValueError('Outputs must stay inside the project')
    if model_dir.exists() or report_dir.exists() or model_dir == report_dir:
        raise ValueError('Use distinct new output directories')
    defaults = base.training_defaults()
    train = base.load_split(defaults['train'])
    val = base.load_split(defaults['val'])
    base.validate_separation(train, val, 'text_tokenized')
    source_path = ROOT/'models/resplit_v1/steps23_v1/manifest.json'
    source = json.loads(source_path.read_text(encoding='utf-8'))
    for key in ('train', 'validation'):
        if source[key]['sha256'] != base.file_hash(defaults['train' if key == 'train' else 'val']):
            raise ValueError('Reference manifest uses another dataset')
    if specs is None:
        specs = []
        for name in REFERENCES:
            metadata = next(m for m in source['models'] if m['model'] == name)
            path = ROOT/metadata['path']
            if base.file_hash(path) != metadata['sha256']:
                raise ValueError('Reference artifact changed')
            reference = joblib.load(path)
            if base.evaluate(val.label, reference.predict(val.text.tolist())) != metadata['validation_metrics']:
                raise ValueError('Reference validation metrics differ')
            specs.append(dict(name='reference_'+name, experiment='reference', profile=metadata['profile'],
                              feature_params=metadata['feature_params'], classifier_params=metadata['classifier_params']))
        specs.extend(controlled_specs())
    folds = group_folds(train) if folds is None else folds
    model_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    prep, inputs = {}, {}
    for profile in sorted({s['profile'] for s in specs}):
        LOG.info('Preparing %s train and validation', profile)
        prep[profile] = CommentPreprocessor(profile)
        inputs[profile] = (prep[profile].fit_transform(train.text.tolist()),
                           prep[profile].transform(val.text.tolist()))
    assignments = train[['id', 'post_id', 'label']].copy()
    for number, (_, held) in enumerate(folds):
        assignments.loc[held, 'fold'] = number
    base.write_csv(report_dir/'train_fold_assignments.csv', assignments)
    models, comparisons, fold_records, threshold_records = [], [], [], []
    y = train.label.to_numpy()
    for count, spec in enumerate(specs, 1):
        LOG.info('Configuration %d/%d: %s', count, len(specs), spec['name'])
        fit_input, val_input = inputs[spec['profile']]
        scores = np.full(len(train), np.nan)
        for number, (fit, held) in enumerate(folds):
            features = _features(spec['feature_params'])
            matrix = features.fit_transform(fit_input.iloc[fit])
            classifier = LinearSVC(**spec['classifier_params'], random_state=42, max_iter=20000)
            classifier.fit(matrix, y[fit])
            scores[held] = classifier.decision_function(features.transform(fit_input.iloc[held]))
            fold_records.append(dict(model=spec['name'], fold=number, train_rows=len(fit), heldout_rows=len(held),
                                     post_overlap=0, **base.evaluate(y[held], (scores[held] >= 0).astype(int))))
        if not np.isfinite(scores).all():
            raise RuntimeError('Missing or invalid out-of-fold predictions')
        selected_threshold, curve = select_threshold(y, scores)
        threshold_records.extend(dict(model=spec['name'], **row) for row in curve)
        LOG.info('OOF threshold %.4f, calibrated Macro-F1 %.4f', selected_threshold['threshold'], selected_threshold['macro_f1'])
        base.write_csv(report_dir/f'oof_predictions_{spec["name"]}.csv', prediction_table(train, scores, selected_threshold['threshold']))
        features = _features(spec['feature_params'])
        classifier = LinearSVC(**spec['classifier_params'], random_state=42, max_iter=20000)
        classifier.fit(features.fit_transform(fit_input), y)
        raw_pipeline = Pipeline([('preprocess', CommentPreprocessor(spec['profile'])),
                                 ('features', features), ('classifier', classifier)])
        val_scores = classifier.decision_function(features.transform(val_input))
        for mode, threshold in [('zero', 0.0), ('oof', selected_threshold['threshold'])]:
            name = spec['name']+'_'+mode
            fitted = DecisionThresholdClassifier(raw_pipeline, threshold)
            predictions = (val_scores >= threshold).astype(int)
            metrics = base.evaluate(val.label, predictions)
            artifact = model_dir/f'{name}.joblib'
            joblib.dump(fitted, artifact, compress=3)
            loaded = joblib.load(artifact)
            if not np.array_equal(loaded.predict(val.text.tolist()), predictions):
                raise RuntimeError('Reload changed predictions')
            if not set(loaded.predict(['', 'ΩΩΩ', 'ko\nđược chửi 👩🏽‍💻', '100k'])) <= {0, 1}:
                raise RuntimeError('Invalid boundary predictions')
            metadata = dict(model=name, backend='sklearn', input_contract='raw_text', threshold=float(threshold),
                            threshold_basis='fixed zero' if mode == 'zero' else 'train grouped out-of-fold scores only',
                            path=artifact.relative_to(ROOT).as_posix(), sha256=base.file_hash(artifact),
                            validation_metrics=metrics, reload_verified=True,
                            profile=spec['profile'], feature_params=spec['feature_params'], classifier_params=spec['classifier_params'])
            models.append(metadata)
            matching = [r for r in fold_records if r['model'] == spec['name']]
            comparisons.append(dict(model=name, experiment=spec['experiment'], threshold=threshold,
                                    fold_macro_mean=float(np.mean([r['macro_f1'] for r in matching])),
                                    fold_macro_std=float(np.std([r['macro_f1'] for r in matching])),
                                    oof_macro_zero=base.evaluate(y, (scores >= 0).astype(int))['macro_f1'],
                                    oof_macro_calibration=selected_threshold['macro_f1'], **metrics))
            predictions_frame = prediction_table(val, val_scores, threshold)
            base.write_csv(report_dir/f'val_predictions_{name}.csv', predictions_frame)
            LOG.info('%s val Macro-F1=%.4f, F1-toxic=%.4f, Recall=%.4f', name, metrics['macro_f1'], metrics['f1_toxic'], metrics['recall_toxic'])
        base.write_csv(report_dir/'validation_comparison.csv', pd.DataFrame(comparisons))
        base.write_csv(report_dir/'group_cv_folds.csv', pd.DataFrame(fold_records))
    winner = max(models, key=lambda m: base.selection_key(m['validation_metrics']))
    base.write_csv(report_dir/'threshold_search.csv', pd.DataFrame(threshold_records))
    manifest = dict(run_id=model_dir.name, completed_at_utc=datetime.now(timezone.utc).isoformat(),
                    train=base.split_summary(defaults['train'], train), validation=base.split_summary(defaults['val'], val),
                    source_manifest_sha256=base.file_hash(source_path),
                    stop_words=STOP_WORDS, folds=3, configurations=len(specs), models=models,
                    best_on_validation=winner['model'], test_evaluated=False,
                    selection_rule='Validation Macro-F1, then F1-toxic, then Recall-toxic; first on exact ties.',
                    threshold_policy='Threshold fitted on pooled train grouped OOF scores; zero is a fixed comparison. Validation selects candidates; test never read.',
                    cv_caveat='Configurations partly originate from previous validation selection; OOF diagnostics and threshold calibration are not unbiased nested-CV estimates.')
    base.write_json(model_dir/'manifest.json', manifest)
    base.write_json(report_dir/'training_summary.json', manifest)
    lines = ['# Tối ưu có kiểm định theo bài viết', '',
             'Giữ nguyên nhãn và bộ chia. Ngưỡng chỉ học từ dự đoán ngoài mẫu trên train.',
             'Tám cấu hình có kiểm soát: hai cách chuẩn hóa × ngram từ 1–2/1–3 × giữ/bỏ danh sách từ dừng nhỏ.',
             'Đặc trưng ký tự giữ nguyên; chỉ bỏ từ dừng ở nhánh từ. Không bỏ phủ định hoặc đại từ chỉ người.', '',
             '| Mô hình | Ngưỡng | Macro-F1 val | F1-toxic val | Recall-toxic val | Macro-F1 fold ± std (ngưỡng 0) |',
             '|---|---:|---:|---:|---:|---|']
    for row in comparisons:
        lines.append(f"| {row['model']} | {row['threshold']:.4f} | {row['macro_f1']:.4f} | {row['f1_toxic']:.4f} | {row['recall_toxic']:.4f} | {row['fold_macro_mean']:.4f} ± {row['fold_macro_std']:.4f} |")
    lines += ['', f"Tốt nhất theo validation: **{winner['model']}**.", '',
              manifest['cv_caveat'],
              'Bảng OOF có gắn dấu hiệu phủ định, mention và câu ngắn để phân tích lỗi; các dấu hiệu này không được dùng làm nhãn.',
              'Chưa đọc test; kết quả test chỉ xuất sau khi khóa lựa chọn.']
    (report_dir/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return manifest


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s', force=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=ROOT/'models/resplit_v1/steps23_v2')
    parser.add_argument('--report-dir', type=Path, default=ROOT/'reports/resplit_v1/steps23_v2')
    args = parser.parse_args(argv)
    try:
        summary = run(args.model_dir, args.report_dir)
        print(json.dumps({k: summary[k] for k in ('best_on_validation', 'configurations', 'test_evaluated')}))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, TypeError) as exc:
        LOG.error('Refinement failed: %s', exc)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
