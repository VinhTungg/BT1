"""Compare word/character TF-IDF + SVM without modifying original baselines.

Saved pipelines accept a DataFrame with text_clean and text_tokenized columns.
Train selects vocabulary/weights, validation selects configurations, test is not read.
"""

from __future__ import annotations

import argparse
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
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import train_baselines as base

ROOT = base.ROOT
INPUT_COLUMNS = ["text_clean", "text_tokenized"]
FAMILIES = ("word_svm_weighted", "char_svm", "word_char_svm")
LOG = logging.getLogger("train_improved_baselines")


def load_inputs(train_path: Path, val_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = base.load_split(train_path)
    val = base.load_split(val_path)
    for path, frame in ((train_path, train), (val_path, val)):
        if "text_clean" not in frame or frame["text_clean"].str.strip().eq("").any():
            raise ValueError(f"{path}: text_clean must exist and contain nonempty strings")
    base.validate_separation(train, val, "text_tokenized")
    return train, val


def build_pipeline(family: str, seed: int = 42) -> Pipeline:
    if family not in FAMILIES:
        raise ValueError(f"Unknown model family: {family}")
    transformers = []
    if family != "char_svm":
        word = TfidfVectorizer(
            tokenizer=str.split, token_pattern=None, lowercase=True,
            ngram_range=(1, 2) if family == "word_char_svm" else (1, 1), min_df=1,
        )
        transformers.append(("word", word, "text_tokenized"))
    if family != "word_svm_weighted":
        char = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=1,
            lowercase=True, sublinear_tf=True,
        )
        transformers.append(("char", char, "text_clean"))
    return Pipeline([
        ("features", ColumnTransformer(transformers, remainder="drop", sparse_threshold=1.0)),
        ("classifier", LinearSVC(random_state=seed, max_iter=10000)),
    ])


def parameter_grid(family: str, quick: bool = False) -> list[dict]:
    if family not in FAMILIES:
        raise ValueError(f"Unknown model family: {family}")
    grid = {
        "classifier__C": [0.1] if quick else [0.1, 1.0, 10.0],
        "classifier__class_weight": ["balanced"] if quick else [
            {0: 1, 1: weight} for weight in (2, 4, 8, 12)
        ] + ["balanced"],
    }
    if family == "word_svm_weighted":
        grid["features__word__ngram_range"] = [(1, 1)] if quick else [(1, 1), (1, 2)]
    return list(ParameterGrid(grid))


def load_reference(manifest_path: Path, train_path: Path, val_path: Path) -> dict:
    reference = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, path in (("train", train_path), ("validation", val_path)):
        if reference[name]["sha256"] != base.file_hash(path):
            raise ValueError(f"Original baseline {name} hash differs; comparisons would be invalid")
    if not reference.get("models"):
        raise ValueError("Original baseline manifest has no models")
    return reference


def artifact_path(manifest_path: Path, metadata: dict) -> Path:
    # Resolve beside manifest first to support moving the handoff to another PC.
    local = manifest_path.parent / f"{metadata['model']}.joblib"
    path = local if local.exists() else Path(metadata["artifact"])
    if base.file_hash(path) != metadata["sha256"]:
        raise ValueError(f"Reference artifact hash differs: {path}")
    return path


def make_group_folds(train: pd.DataFrame, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if train["post_id"].nunique() < 3:
        raise ValueError("At least 3 training posts are required for group diagnostics")
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    return list(splitter.split(train[INPUT_COLUMNS], train["label"], groups=train["post_id"]))


def group_diagnostics(model: str, pipeline: Pipeline, train: pd.DataFrame, folds: list, *, original: bool = False) -> list[dict]:
    records = []
    for fold, (fit_indices, held_indices) in enumerate(folds, 1):
        fit, held = train.iloc[fit_indices], train.iloc[held_indices]
        overlap = set(fit["post_id"]) & set(held["post_id"])
        if overlap:
            raise RuntimeError("Group fold leakage")
        record = {
            "model": model, "fold": fold, "train_rows": len(fit), "heldout_rows": len(held),
            "train_posts": fit["post_id"].nunique(), "heldout_posts": held["post_id"].nunique(),
            "heldout_toxic": int(held["label"].sum()), "post_overlap": 0,
        }
        try:
            if set(fit["label"]) != {0, 1} or set(held["label"]) != {0, 1}:
                raise ValueError("A group fold lacks one of the classes")
            x_fit = fit["text_tokenized"].tolist() if original else fit[INPUT_COLUMNS]
            x_held = held["text_tokenized"].tolist() if original else held[INPUT_COLUMNS]
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                estimator = clone(pipeline).fit(x_fit, fit["label"])
            record.update(status="ok", error="", **base.evaluate(held["label"], estimator.predict(x_held)))
        except (ValueError, ConvergenceWarning) as exc:
            record.update(status="failed", error=str(exc))
        records.append(record)
    return records


def run_experiments(
    train_path: Path, val_path: Path, baseline_manifest: Path, model_dir: Path, report_dir: Path,
    *, seed: int = 42, quick: bool = False,
) -> dict:
    if model_dir.resolve() == baseline_manifest.parent.resolve():
        raise ValueError("Use a separate model directory to preserve original baselines")
    train, val = load_inputs(train_path, val_path)
    reference = load_reference(baseline_manifest, train_path, val_path)
    reference_hash = base.file_hash(baseline_manifest)
    folds = make_group_folds(train, seed)
    x_train, x_val = train[INPUT_COLUMNS], val[INPUT_COLUMNS]
    y_train, y_val = train["label"].to_numpy(), val["label"].to_numpy()
    reference_best = reference["best_baseline_on_validation"]
    reference_meta = next(item for item in reference["models"] if item["model"] == reference_best)
    reference_pipeline = joblib.load(artifact_path(baseline_manifest, reference_meta))
    if base.evaluate(y_val, reference_pipeline.predict(val["text_tokenized"].tolist())) != reference_meta["validation_metrics"]:
        raise ValueError("Reference model no longer reproduces its saved validation metrics")
    LOG.info("Train %d rows, validation %d; fixed grid: %d trials", len(train), len(val), sum(len(parameter_grid(f, quick)) for f in FAMILIES))
    selected, trials = [], []
    for family in FAMILIES:
        best = None
        for trial, params in enumerate(parameter_grid(family, quick), 1):
            pipeline = build_pipeline(family, seed).set_params(**params)
            start = time.perf_counter()
            record = {"model": family, "trial": trial, "params": json.dumps(params, sort_keys=True)}
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    pipeline.fit(x_train, y_train)
                predictions = pipeline.predict(x_val)
                metrics = base.evaluate(y_val, predictions)
                record.update(status="ok", error="", **metrics)
                if best is None or base.selection_key(metrics) > base.selection_key(best["metrics"]):
                    best = {
                        "model": family, "pipeline": pipeline, "params": params,
                        "metrics": metrics, "predictions": predictions, "selected_trial": trial,
                    }
            except (ValueError, ConvergenceWarning) as exc:
                record.update(status="failed", error=str(exc))
                LOG.warning("%s trial %d: %s", family, trial, exc)
            record["fit_predict_seconds"] = round(time.perf_counter() - start, 6)
            trials.append(record)
        if best is None:
            report_dir.mkdir(parents=True, exist_ok=True)
            base.write_csv(report_dir / "validation_trials.csv", pd.DataFrame(trials))
            raise ValueError(f"All trials failed for {family}; see validation_trials.csv")
        best["configurations_tried"] = len(parameter_grid(family, quick))
        selected.append(best)
        LOG.info("%s: Macro-F1 %.4f; F1-toxic %.4f; FP %d; FN %d", family, best["metrics"]["macro_f1"], best["metrics"]["f1_toxic"], best["metrics"]["fp"], best["metrics"]["fn"])

    LOG.info("Running 3-fold group diagnostics on selected configurations and original best")
    diagnostic_rows = group_diagnostics(reference_best, reference_pipeline, train, folds, original=True)
    for best in selected:
        diagnostic_rows.extend(group_diagnostics(best["model"], best["pipeline"], train, folds))
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "quick" if quick else "full", "seed": seed,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
        "train": base.split_summary(train_path, train), "validation": base.split_summary(val_path, val),
        "baseline_manifest": str(baseline_manifest.resolve()), "baseline_manifest_sha256": reference_hash,
        "input_contract": {"type": "pandas.DataFrame", "columns": INPUT_COLUMNS, "word_input": "text_tokenized", "char_input": "text_clean"},
        "label_mapping": {"0": "non-toxic", "1": "toxic"},
        "selection_rule": reference["selection_rule"],
        "fit_policy": "Fit on train only; select on validation; fixed SVM decision threshold 0; test never read; original baselines unchanged.",
        "group_cv_caveat": "3 folds from train posts only, configurations already selected on validation. Post-selection diagnostics, not unbiased nested-CV estimates and not test scores.",
        "models": [],
    }
    comparison = [
        {"model": item["model"], "experiment": "original", **item["validation_metrics"], "params": json.dumps(item["params"], sort_keys=True)}
        for item in reference["models"]
    ]
    for best in selected:
        name = best["model"]
        path = model_dir / f"{name}.joblib"
        temporary = path.with_name(path.name + ".tmp")
        joblib.dump(best["pipeline"], temporary, compress=3)
        reloaded = joblib.load(temporary)
        if not np.array_equal(reloaded.predict(x_val), best["predictions"]):
            raise RuntimeError(f"Reload prediction mismatch for {name}")
        temporary.replace(path)
        feature_counts = {
            branch: len(vectorizer.vocabulary_)
            for branch, vectorizer, _ in reloaded.named_steps["features"].transformers_
            if branch in ("word", "char")
        }
        summary["models"].append({
            "model": name, "artifact": str(path.resolve()), "sha256": base.file_hash(path),
            "params": best["params"], "selected_trial": best["selected_trial"],
            "configurations_tried": best["configurations_tried"], "validation_metrics": best["metrics"],
            "vocabulary_sizes": feature_counts, "reload_verified": True,
        })
        comparison.append({"model": name, "experiment": "improved", **best["metrics"], "params": json.dumps(best["params"], sort_keys=True)})
        columns = [column for column in ("id", "post_id", "text", *INPUT_COLUMNS) if column in val]
        predictions = val[columns].copy()
        predictions["label"] = y_val
        predictions["prediction"] = best["predictions"]
        predictions["outcome"] = [
            ("TP" if actual == 1 else "TN") if actual == predicted else ("FN" if actual == 1 else "FP")
            for actual, predicted in zip(y_val, best["predictions"])
        ]
        base.write_csv(report_dir / f"val_predictions_{name}.csv", predictions)
    winner = max((row for row in comparison if row["model"] != "dummy"), key=base.selection_key)
    summary["best_model_on_validation"] = winner["model"]
    summary["delta_vs_original_best"] = {
        metric: winner[metric] - reference_meta["validation_metrics"][metric]
        for metric in ("macro_f1", "f1_toxic", "precision_toxic", "recall_toxic", "fp", "fn")
    }
    diagnostic = pd.DataFrame(diagnostic_rows)
    cv_rows = []
    for model in (reference_best, *FAMILIES):
        valid = diagnostic[(diagnostic["model"] == model) & (diagnostic["status"] == "ok")]
        cv_rows.append({
            "model": model, "successful_folds": len(valid), "failed_folds": 3 - len(valid),
            "macro_f1_mean": float(valid["macro_f1"].mean()) if len(valid) else None,
            "macro_f1_std": float(valid["macro_f1"].std(ddof=0)) if len(valid) else None,
            "f1_toxic_mean": float(valid["f1_toxic"].mean()) if len(valid) else None,
        })
    summary["group_diagnostics"] = cv_rows
    winner_cv = next(row for row in cv_rows if row["model"] == winner["model"])
    reference_cv = next(row for row in cv_rows if row["model"] == reference_best)
    summary["group_cv_macro_f1_delta_vs_original"] = (
        winner_cv["macro_f1_mean"] - reference_cv["macro_f1_mean"]
        if winner_cv["macro_f1_mean"] is not None and reference_cv["macro_f1_mean"] is not None else None
    )
    summary["generalization_improvement_confirmed"] = False
    summary["handoff_recommendation"] = (
        "Keep the four original baselines for the required comparison; hand off the validation winner "
        "as an additional candidate. Group diagnostics are exploratory; confirm performance with final test evaluation."
    )
    fold_ids = train[["id", "post_id", "label"]].copy()
    fold_ids["heldout_fold"] = 0
    for number, (_, held_indices) in enumerate(folds, 1):
        fold_ids.loc[fold_ids.index[held_indices], "heldout_fold"] = number
    # Validate source and reference fingerprints before publishing final manifests.
    for name, path in (("train", train_path), ("validation", val_path)):
        if base.file_hash(path) != reference[name]["sha256"]:
            raise RuntimeError(f"{name} changed while training")
    if base.file_hash(baseline_manifest) != reference_hash:
        raise RuntimeError("Original baseline manifest changed while training")
    base.write_csv(report_dir / "validation_trials.csv", pd.DataFrame(trials))
    base.write_csv(report_dir / "validation_comparison.csv", pd.DataFrame(comparison))
    base.write_csv(report_dir / "group_cv_folds.csv", diagnostic)
    base.write_csv(report_dir / "group_cv_summary.csv", pd.DataFrame(cv_rows))
    base.write_csv(report_dir / "train_fold_assignments.csv", fold_ids)
    base.write_json(report_dir / "training_summary.json", summary)
    base.write_json(model_dir / "manifest.json", summary)
    lines = [
        "# Thí nghiệm cải tiến ML baseline", "",
        f"- Chế độ: {summary['mode']}; seed {seed}; {len(trials)} cấu hình mới.",
        f"- Train {len(train)} câu; validation {len(val)} câu. Không đánh giá test.",
        "- Chọn Macro-F1, rồi F1-toxic, Recall-toxic; hòa hoàn toàn giữ cấu hình đầu tiên.",
        "- Ngưỡng quyết định SVM giữ ở 0. Không thu thập thêm hoặc sửa nhãn.", "",
        "| Mô hình | Nhánh | Accuracy | Precision toxic | Recall toxic | F1 toxic | Macro-F1 | FP | FN |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison:
        scores = " | ".join(f"{row[k]:.4f}" for k in ("accuracy", "precision_toxic", "recall_toxic", "f1_toxic", "macro_f1"))
        lines.append(f"| {row['model']} | {row['experiment']} | {scores} | {row['fp']} | {row['fn']} |")
    delta = summary["delta_vs_original_best"]
    lines.extend([
        "", f"Tốt nhất trên validation: **{winner['model']}**.",
        f"So với {reference_best}: Macro-F1 {delta['macro_f1']:+.4f}, F1-toxic {delta['f1_toxic']:+.4f}, FP {delta['fp']:+d}, FN {delta['fn']:+d}.",
        "", "## Kiểm tra bổ sung trên ba fold theo bài viết trong train", "",
        "Các cấu hình đã được chọn bằng validation; đây là chẩn đoán sau lựa chọn, không phải nested CV hay kết quả test.", "",
        "| Mô hình | Fold thành công | Macro-F1 trung bình | Độ lệch chuẩn | F1-toxic trung bình |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in cv_rows:
        scores = " | ".join(f"{row[key]:.4f}" if row[key] is not None else "N/A" for key in ("macro_f1_mean", "macro_f1_std", "f1_toxic_mean"))
        lines.append(f"| {row['model']} | {row['successful_folds']}/3 | {scores} |")
    group_delta = summary["group_cv_macro_f1_delta_vs_original"]
    if group_delta is not None:
        lines.extend(["", f"Chênh lệch Macro-F1 trung bình ba fold của ứng viên tốt nhất so với baseline gốc: {group_delta:+.4f}."])
        if group_delta < 0:
            lines.append("Ứng viên thắng trên validation nhưng chưa được kiểm tra nhóm ủng hộ; chưa kết luận khả năng khái quát tốt hơn.")
    lines.extend(["", "Giữ bốn baseline gốc cho bảng thực nghiệm bắt buộc, bàn giao ứng viên mới như mô hình bổ sung."])
    lines.extend(["", "## Cấu hình tốt nhất từng nhánh", ""])
    for best in selected:
        lines.append(f"- `{best['model']}`: `{json.dumps(best['params'], sort_keys=True)}`")
    lines.extend([
        "", "Validation nhỏ và đã được dùng nhiều lần để chọn cấu hình; chưa thể khẳng định mức tăng sẽ giữ nguyên trên dữ liệu mới.",
        "Pipeline mới nhận DataFrame chứa `text_clean` và `text_tokenized`, khác đầu vào danh sách chuỗi của baseline gốc.",
        "Bàn giao cả baseline gốc và thí nghiệm mới cho Người 4; dùng test độc lập khi chốt đánh giá.", "",
    ])
    (report_dir / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")
    LOG.info("Best validation model: %s; Macro-F1 delta %+.4f", winner["model"], delta["macro_f1"])
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--val", type=Path)
    parser.add_argument("--baseline-manifest", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        defaults = base.training_defaults() if any(v is None for v in (args.train, args.val, args.baseline_manifest, args.model_dir, args.report_dir)) else {}
        train_path, val_path = args.train or defaults['train'], args.val or defaults['val']
        baseline_manifest = args.baseline_manifest or defaults['baseline_models']/'manifest.json'
        model_dir = args.model_dir or defaults['improved_models']
        report_dir = args.report_dir or defaults['improved_reports']
        if args.quick:
            if args.model_dir is None:
                model_dir /= 'quick'
            if args.report_dir is None:
                report_dir /= 'quick'
        run_experiments(train_path, val_path, baseline_manifest, model_dir, report_dir, seed=args.seed, quick=args.quick)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        LOG.error("Experiments failed: %s", exc)
        return 1
    LOG.info("Models: %s; reports: %s", model_dir, report_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
