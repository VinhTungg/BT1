"""Optimize steps 2/3 on the active train/validation split; never open test.

Run: python scripts/optimize_text_features.py
Outputs include raw-text pipelines, a controlled comparison, OOV diagnostics,
all trials, and a validation report. Existing outputs are never overwritten.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import train_baselines as base
from src.text_features import CommentPreprocessor, PROFILE_VERSION, PROFILES, build_features

LOG = logging.getLogger("optimize_text_features")


def feature_configs(quick=False):
    for kind in ("bow", "tfidf"):
        for ngrams, minimum in itertools.product(
            [(1, 1)] if quick else [(1, 1), (1, 2)], [1] if quick else [1, 2],
        ):
            yield dict(kind=kind, ngram_range=ngrams, min_df=minimum)
    for char_range in ([(3, 5)] if quick else [(2, 5), (3, 5)]):
        yield dict(kind="char", char_range=char_range, min_df=2)
    for weight in ([1.0] if quick else [0.5, 1.0]):
        yield dict(kind="word_char", ngram_range=(1, 2), min_df=2, char_weight=weight)


def classifier_configs(kind, quick=False):
    if kind in ("bow", "tfidf"):
        for alpha in ([0.1] if quick else [0.1, 1.0]):
            yield "nb", dict(alpha=alpha)
    for c in ([0.1] if quick else [0.1, 1.0, 10.0]):
        yield "svm", dict(C=c, class_weight="balanced")


def diagnostics(features, frame):
    matrix = features.transform(frame)
    result = {"features": int(matrix.shape[1]),
              "zero_vectors": int(np.count_nonzero(matrix.getnnz(axis=1) == 0))}
    branches = features.named_transformers_
    if "word" in branches:
        vectorizer = branches["word"]
        tokens = [token for text in frame.text_tokenized for token in vectorizer.build_analyzer()(text)]
        unseen = sum(token not in vectorizer.vocabulary_ for token in tokens)
        result.update(word_features=len(vectorizer.vocabulary_), oov_tokens=unseen,
                      total_tokens=len(tokens), oov_fraction=unseen / len(tokens) if tokens else 0.0)
    return result


def normalization_overlap(train, val):
    """Report collisions created by fixed rules, without relabeling/dropping rows.

    Original IDs/posts/text are checked before preprocessing. Deterministic
    feature equivalence (e.g. ko/không) is not a fit on validation data.
    """
    result = {}
    for column in ("text_clean", "text_tokenized"):
        fit_text = set(train[column].str.casefold().str.split().str.join(" "))
        held_text = val[column].str.casefold().str.split().str.join(" ")
        result[column] = int(held_text.isin(fit_text).sum())
    return result


def run(model_dir: Path, report_dir: Path, *, quick=False, promote=True):
    model_dir, report_dir = model_dir.resolve(), report_dir.resolve()
    for directory in (model_dir, report_dir):
        if not directory.is_relative_to(ROOT.resolve()):
            raise ValueError("Optimization outputs must stay inside the project")
    if model_dir.exists() or report_dir.exists():
        raise ValueError("Output already exists; choose new directories to preserve completed runs")
    if model_dir.resolve() == report_dir.resolve():
        raise ValueError("Model and report directories must differ")
    defaults = base.training_defaults()  # Resolves/verifies train and val only.
    train_path, val_path = defaults["train"], defaults["val"]
    train, val = base.load_split(train_path), base.load_split(val_path)
    for frame in (train, val):
        if "text" not in frame or frame.text.str.strip().eq("").any():
            raise ValueError("Raw text is required for consistent preprocessing")
    base.validate_separation(train, val, "text_tokenized")
    baseline_path = defaults["baseline_models"] / "manifest.json"
    reference = json.loads(baseline_path.read_text(encoding="utf-8"))
    for key, path in (("train", train_path), ("validation", val_path)):
        if reference[key]["sha256"] != base.file_hash(path):
            raise ValueError("Reference baseline uses different data")
    reference_models = reference["models"]
    original_best = max((m for m in reference_models if m["model"] != "dummy"),
                        key=lambda m: base.selection_key(m["validation_metrics"]))
    y_train, y_val = train.label.to_numpy(), val.label.to_numpy()
    model_dir.mkdir(parents=True, exist_ok=False)
    report_dir.mkdir(parents=True, exist_ok=False)
    trials, selected, controlled = [], [], []
    overlaps = {}
    total = sum(len(list(classifier_configs(c["kind"], quick))) for c in feature_configs(quick)) * len(PROFILES)
    for profile in PROFILES:
        LOG.info("Preprocessing %s: train=%d, validation=%d", profile, len(train), len(val))
        preprocessor = CommentPreprocessor(profile)
        x_train = preprocessor.fit_transform(train.text.tolist())
        x_val = preprocessor.transform(val.text.tolist())
        overlaps[profile] = normalization_overlap(x_train, x_val)
        best = {}
        for feature_params in feature_configs(quick):
            features = build_features(**feature_params)
            fit_matrix = features.fit_transform(x_train)
            val_matrix = features.transform(x_val)
            for algorithm, classifier_params in classifier_configs(feature_params["kind"], quick):
                family = feature_params["kind"] + "_" + algorithm
                name = profile + "_" + family
                classifier = (MultinomialNB(**classifier_params) if algorithm == "nb" else
                              LinearSVC(**classifier_params, random_state=42, max_iter=20000))
                record = dict(model=name, profile=profile, family=family,
                              feature_params=json.dumps(feature_params, sort_keys=True),
                              classifier_params=json.dumps(classifier_params, sort_keys=True),
                              features=int(fit_matrix.shape[1]))
                start = time.perf_counter()
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", ConvergenceWarning)
                        classifier.fit(fit_matrix, y_train)
                    predicted = classifier.predict(val_matrix)
                    metrics = base.evaluate(y_val, predicted)
                    record.update(status="ok", error="", **metrics)
                    if family not in best or base.selection_key(metrics) > base.selection_key(best[family]["validation_metrics"]):
                        best[family] = dict(model=name, profile=profile, family=family,
                                            validation_metrics=metrics, features=features,
                                            classifier=classifier, predictions=predicted,
                                            feature_params=feature_params, classifier_params=classifier_params)
                    if (feature_params["kind"] in ("bow", "tfidf") and
                        feature_params["ngram_range"] == (1, 1) and feature_params["min_df"] == 1 and
                        classifier_params.get("alpha", classifier_params.get("C")) == 0.1):
                        controlled.append(record.copy())
                except (ValueError, ConvergenceWarning) as exc:
                    record.update(status="failed", error=str(exc))
                record["fit_predict_seconds"] = round(time.perf_counter() - start, 4)
                trials.append(record)
                base.write_csv(report_dir / "validation_trials.csv", pd.DataFrame(trials))
                LOG.info("%d/%d %s %s Macro-F1=%s", len(trials), total, name, record["status"], record.get("macro_f1", "failed"))
        if len(best) != 6:
            raise ValueError(f"A feature family failed completely for {profile}; see validation_trials.csv")
        for item in best.values():
            pipeline = Pipeline([("preprocess", CommentPreprocessor(profile)),
                                 ("features", item["features"]), ("classifier", item["classifier"])])
            artifact = model_dir / f"{item['model']}.joblib"
            joblib.dump(pipeline, artifact, compress=3)
            loaded = joblib.load(artifact)
            if not np.array_equal(loaded.predict(val.text.tolist()), item["predictions"]):
                raise RuntimeError("Saved raw-text pipeline changed validation predictions")
            if not set(loaded.predict(["", "ΩΩΩ", "ko\nđược chửi 👩🏽‍💻", "100k"])) <= {0, 1}:
                raise RuntimeError("Invalid edge-case predictions")
            metadata = {k: v for k, v in item.items() if k not in ("features", "classifier", "predictions")}
            metadata.update(path=artifact.relative_to(ROOT).as_posix(), sha256=base.file_hash(artifact),
                            input_contract="raw_text", reload_verified=True,
                            validation_features=diagnostics(item["features"], x_val))
            selected.append(metadata)
            predictions = val[["id", "post_id", "text", "label"]].copy()
            predictions["prediction"] = item["predictions"]
            predictions["outcome"] = [
                ("TP" if actual else "TN") if actual == predicted else ("FN" if actual else "FP")
                for actual, predicted in zip(y_val, item["predictions"])
            ]
            base.write_csv(report_dir / f"val_predictions_{item['model']}.csv", predictions)
    winner = max(selected, key=lambda m: base.selection_key(m["validation_metrics"]))
    can_promote = not quick and base.selection_key(winner["validation_metrics"]) > base.selection_key(original_best["validation_metrics"])
    summary = dict(
        run_id=model_dir.name, completed_at_utc=datetime.now(timezone.utc).isoformat(),
        profile_version=PROFILE_VERSION, seed=42, quick=quick,
        versions=dict(python=platform.python_version(), sklearn=sklearn.__version__, pandas=pd.__version__),
        train=base.split_summary(train_path, train), validation=base.split_summary(val_path, val),
        baseline_manifest_sha256=base.file_hash(baseline_path),
        preprocessing_sha256=base.file_hash(ROOT / "src/preprocessing.py"),
        text_features_sha256=base.file_hash(ROOT / "src/text_features.py"),
        experiment_script_sha256=base.file_hash(Path(__file__)),
        label_policy="Dataset labels accepted and finalized by the group on 2026-11-02.",
        selection_rule=reference["selection_rule"],
        fit_policy="Fit features and classifiers on train; select on validation only; test never opened.",
        trials=len(trials), failed_trials=sum(r["status"] != "ok" for r in trials),
        best_on_validation=winner["model"], reference_best=original_best,
        models=selected, test_evaluated=False, promoted=bool(promote and can_promote),
        normalization_overlap_on_validation=overlaps,
    )
    base.write_json(model_dir / "manifest.json", summary)
    base.write_json(report_dir / "training_summary.json", summary)
    comparison = [dict(model=m["model"], experiment="reference", **m["validation_metrics"])
                  for m in reference_models]
    comparison += [dict(model=m["model"], experiment="steps23", **m["validation_metrics"],
                        **m["validation_features"]) for m in selected]
    base.write_csv(report_dir / "validation_comparison.csv", pd.DataFrame(comparison))
    base.write_csv(report_dir / "controlled_comparison.csv", pd.DataFrame(controlled))
    lines = ["# Tối ưu bước 2 và 3 trên validation", "",
             "Nhãn đã được nhóm chốt. Giữ nguyên ID, nhãn và phân chia dữ liệu.",
             f"Train {len(train)}, validation {len(val)}. Test không được đọc trong lượt này.",
             f"Lưới cố định: {len(trials)} cấu hình, {summary['failed_trials']} cấu hình lỗi.",
             "Chọn Macro-F1; hòa điểm xét F1-toxic, Recall-toxic, rồi cấu hình đầu tiên.", "",
             "| Mô hình | Accuracy | Precision toxic | Recall toxic | F1-toxic | Macro-F1 |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in comparison:
        values = " | ".join(f"{row[k]:.4f}" for k in ("accuracy", "precision_toxic", "recall_toxic", "f1_toxic", "macro_f1"))
        lines.append(f"| {row['model']} | {values} |")
    delta = winner["validation_metrics"]["macro_f1"] - original_best["validation_metrics"]["macro_f1"]
    lines += ["", f"Tốt nhất: **{winner['model']}**; chênh Macro-F1 so với baseline: **{delta * 100:+.2f} điểm phần trăm**.",
              f"Đặc trưng: `{json.dumps(winner['feature_params'], ensure_ascii=False)}`.",
              f"Classifier: `{json.dumps(winner['classifier_params'], ensure_ascii=False)}`.", "",
              "`preserve`: chuẩn hóa an toàn, giữ phủ định/dấu câu/emoji, gộp placeholder thành một token.",
              "`slang`: thêm từ điển cố định cho ko/k0/khum/hok/khong/đc/thik/cx/rùi/nma; không thay k, r, đk hay 100k.",
              "BoW và TF-IDF thử cùng ngram/min_df. TF-IDF dùng sublinear TF, IDF làm mịn và L2.",
              "Nhánh ký tự dùng char_wb; nhánh kết hợp dùng từ + ký tự với trọng số cố định trong lưới.",
              "`controlled_comparison.csv` giữ unigram, min_df=1, alpha=0.1 hoặc C=0.1/balanced để tách ảnh hưởng đặc trưng và tiền xử lý.",
              "Từ vựng và IDF chỉ fit train. Artifact nhận văn bản thô và chứa toàn bộ tiền xử lý.",
              f"Số câu validation trùng biểu diễn sau quy tắc chuẩn hóa: `{json.dumps(overlaps)}`. Đây là va chạm do ánh xạ cố định; không gộp câu, không sửa nhãn và không học từ validation.",
              "Điểm validation đã được dùng để chọn cấu hình; chưa phải kết quả đánh giá cuối trên test.",
              "Không dùng điểm test trong các lượt cũ để chọn cấu hình của lượt này."]
    report = "\n".join(lines) + "\n"
    (report_dir / "REPORT.md").write_text(report, encoding="utf-8")
    if promote and can_promote:
        pointer = ROOT / "models/current_run.json"
        previous = json.loads(pointer.read_text(encoding="utf-8"))
        base.write_json(report_dir / "previous_current_run.json", previous)
        active = json.loads((ROOT / "data/current_dataset.json").read_text(encoding="utf-8"))
        current = dict(run_id=summary["run_id"], dataset_id=active["dataset_id"], status="validation_selected",
                       dataset_manifest=active["manifest"], dataset_manifest_sha256=active["manifest_sha256"],
                       train_path=train_path.relative_to(ROOT).as_posix(), train_sha256=base.file_hash(train_path),
                       val_sha256=base.file_hash(val_path), train_samples=len(train), validation_samples=len(val),
                       best_on_validation=winner["model"], validation_metrics=winner["validation_metrics"],
                       models=selected, test_evaluated=False, label_policy=summary["label_policy"],
                       feature_optimization_manifest=(model_dir / "manifest.json").relative_to(ROOT).as_posix(),
                       validation_report=(report_dir / "REPORT.md").relative_to(ROOT).as_posix())
        base.write_json(pointer, current)
        (ROOT / "reports/TRAINING_CURRENT.md").write_text(report, encoding="utf-8")
    return summary


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-promote", action="store_true")
    args = parser.parse_args(argv)
    name = "steps23_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    try:
        result = run(args.model_dir or ROOT / "models" / name,
                     args.report_dir or ROOT / "reports" / name,
                     quick=args.quick, promote=not args.no_promote)
        print(json.dumps({k: result[k] for k in ("run_id", "best_on_validation", "trials", "failed_trials", "promoted", "test_evaluated")}, ensure_ascii=False))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        LOG.error("Optimization failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
