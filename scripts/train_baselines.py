"""Train four text baselines on train; select by validation, never read test.

Input to saved pipelines: an iterable of already segmented Vietnamese strings
(the text_tokenized column). No project-specific objects are needed to load them.
"""

from __future__ import annotations

import argparse
import hashlib
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
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import ParameterGrid
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("bow_nb", "bow_svm", "tfidf_nb", "tfidf_svm")
LOG = logging.getLogger("train_baselines")


def load_split(path: Path, text_column: str = "text_tokenized") -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    missing = {"id", "post_id", "label", text_column} - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path}: dataset is empty")
    for column in ("id", "post_id", text_column):
        frame[column] = frame[column].str.strip()
        if frame[column].eq("").any():
            raise ValueError(f"{path}: empty {column}")
    frame["label"] = frame["label"].str.strip()
    if not frame["label"].isin(["0", "1"]).all():
        raise ValueError(f"{path}: labels must be 0 or 1")
    frame["label"] = frame["label"].astype(int)
    if set(frame["label"]) != {0, 1}:
        raise ValueError(f"{path}: both classes 0 and 1 are required")
    if frame["id"].duplicated().any():
        raise ValueError(f"{path}: duplicate id")
    return frame


def validate_separation(train: pd.DataFrame, val: pd.DataFrame, text_column: str) -> None:
    for column in ("id", "post_id"):
        overlap = set(train[column]) & set(val[column])
        if overlap:
            raise ValueError(f"Train/validation leakage: {len(overlap)} shared {column}")
    # Check both original clean text, if provided, and actual model input.
    for column in dict.fromkeys(("text_clean", text_column)):
        if column in train and column in val:
            normalize = lambda series: set(series.str.lower().str.split().str.join(" "))
            overlap = normalize(train[column]) & normalize(val[column])
            if overlap:
                raise ValueError(f"Train/validation leakage: duplicate {column}")


def parameter_grid(family: str, quick: bool = False) -> list[dict]:
    grid = {
        "vectorizer__ngram_range": [(1, 1)] if quick else [(1, 1), (1, 2)],
        "vectorizer__min_df": [1] if quick else [1, 2],
    }
    if family.endswith("_nb"):
        grid["classifier__alpha"] = [1.0] if quick else [0.1, 0.5, 1.0]
    else:
        grid["classifier__C"] = [1.0] if quick else [0.1, 1.0, 10.0]
        grid["classifier__class_weight"] = ["balanced"] if quick else [None, "balanced"]
    return list(ParameterGrid(grid))


def build_pipeline(family: str, seed: int = 42) -> Pipeline:
    if family == "dummy":
        return Pipeline([("classifier", DummyClassifier(strategy="most_frequent"))])
    if family not in FAMILIES:
        raise ValueError(f"Unknown model family: {family}")
    vectorizer = TfidfVectorizer if family.startswith("tfidf") else CountVectorizer
    # str.split preserves pre-segmented words, single-character words, and emoji.
    # A built-in callable also keeps joblib artifacts independent of this module.
    features = vectorizer(tokenizer=str.split, token_pattern=None, lowercase=True)
    classifier = (
        MultinomialNB() if family.endswith("_nb")
        else LinearSVC(random_state=seed, max_iter=10000)
    )
    return Pipeline([("vectorizer", features), ("classifier", classifier)])


def evaluate(y_true, y_pred) -> dict:
    report = classification_report(
        y_true, y_pred, labels=[0, 1], output_dict=True, zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_toxic": float(report["1"]["precision"]),
        "recall_toxic": float(report["1"]["recall"]),
        "f1_toxic": float(report["1"]["f1-score"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "predicted_toxic": int(np.sum(np.asarray(y_pred) == 1)),
    }


def selection_key(metrics: dict) -> tuple:
    # Compare full precision; exact ties retain the first ParameterGrid entry.
    return tuple(metrics[name] for name in ("macro_f1", "f1_toxic", "recall_toxic"))


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_defaults() -> dict[str, Path]:
    """Read the active dataset without using test data during training."""
    defaults = {
        'train': ROOT/'data/splits/train.csv', 'val': ROOT/'data/splits/val.csv',
        'baseline_models': ROOT/'models/baselines', 'baseline_reports': ROOT/'reports/baselines',
        'improved_models': ROOT/'models/baseline_improvements',
        'improved_reports': ROOT/'reports/baseline_improvements',
    }
    pointer = ROOT/'data/current_dataset.json'
    if not pointer.exists():
        return defaults
    active = json.loads(pointer.read_text(encoding='utf-8'))
    manifest_path = ROOT/active['manifest']
    if file_hash(manifest_path) != active['manifest_sha256']:
        raise ValueError('Active dataset manifest fingerprint changed')
    dataset = json.loads(manifest_path.read_text(encoding='utf-8'))
    for name in ('train', 'val'):
        metadata = dataset['splits'][name]
        path = (ROOT/metadata['path']).resolve()
        if not path.is_relative_to(ROOT.resolve()) or file_hash(path) != metadata['sha256']:
            raise ValueError(f'Active {name} dataset changed or is outside the project')
        defaults[name] = path
    for name in ('baseline_models', 'baseline_reports', 'improved_models', 'improved_reports'):
        path = (ROOT/active['outputs'][name]).resolve()
        if not path.is_relative_to(ROOT.resolve()):
            raise ValueError('Active default output must be within the project')
        defaults[name] = path
    return defaults


def write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def split_summary(path: Path, frame: pd.DataFrame) -> dict:
    return {
        "path": str(path.resolve()), "sha256": file_hash(path),
        "rows": len(frame), "posts": int(frame["post_id"].nunique()),
        "labels": {str(label): int(count) for label, count in frame["label"].value_counts().items()},
    }


def run_training(
    train_path: Path, val_path: Path, model_dir: Path, report_dir: Path,
    *, text_column: str = "text_tokenized", seed: int = 42, quick: bool = False,
) -> dict:
    train = load_split(train_path, text_column)
    val = load_split(val_path, text_column)
    validate_separation(train, val, text_column)
    x_train, x_val = train[text_column].tolist(), val[text_column].tolist()
    y_train, y_val = train["label"].to_numpy(), val["label"].to_numpy()
    LOG.info("Train: %d rows; validation: %d rows", len(train), len(val))
    trials, selected = [], []
    # Finish model selection before publishing artifacts.
    for family in (*FAMILIES, "dummy"):
        best = None
        configurations = [{}] if family == "dummy" else parameter_grid(family, quick)
        for index, params in enumerate(configurations, 1):
            pipeline = build_pipeline(family, seed).set_params(**params)
            start = time.perf_counter()
            record = {"model": family, "trial": index, "params": json.dumps(params, sort_keys=True)}
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    pipeline.fit(x_train, y_train)
                predictions = pipeline.predict(x_val)
                metrics = evaluate(y_val, predictions)
                record.update(status="ok", error="", **metrics)
                if best is None or selection_key(metrics) > selection_key(best["metrics"]):
                    best = {
                        "model": family, "pipeline": pipeline, "params": params,
                        "metrics": metrics, "predictions": predictions, "selected_trial": index,
                    }
            except (ValueError, ConvergenceWarning) as exc:
                record.update(status="failed", error=str(exc))
                LOG.warning("%s trial %d failed: %s", family, index, exc)
            record["fit_predict_seconds"] = round(time.perf_counter() - start, 6)
            trials.append(record)
        if best is None:
            # Preserve diagnostics even when no candidate in a family succeeds.
            report_dir.mkdir(parents=True, exist_ok=True)
            write_csv(report_dir / "validation_trials.csv", pd.DataFrame(trials))
            raise ValueError(f"All configurations failed for {family}; inspect validation_trials.csv")
        best["configurations_tried"] = len(configurations)
        selected.append(best)
        LOG.info(
            "%s: %d configurations; val Macro-F1=%.4f, F1-toxic=%.4f",
            family, len(configurations), best["metrics"]["macro_f1"], best["metrics"]["f1_toxic"],
        )

    best_baseline = max(
        (model for model in selected if model["model"] != "dummy"),
        key=lambda model: selection_key(model["metrics"]),
    )["model"]

    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    versions = {
        "python": platform.python_version(), "scikit-learn": sklearn.__version__,
        "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__,
    }
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "quick" if quick else "full", "seed": seed, "versions": versions,
        "best_baseline_on_validation": best_baseline,
        "train": split_summary(train_path, train), "validation": split_summary(val_path, val),
        "text_column": text_column, "label_mapping": {"0": "non-toxic", "1": "toxic"},
        "selection_rule": "Highest validation macro_f1, then f1_toxic, then recall_toxic; exact ties keep first trial.",
        "inference_input": "Iterable of Vietnamese strings segmented like text_tokenized; vectorizer lowercases and splits on whitespace.",
        "fit_policy": "All artifacts fitted on train only. Validation selects configurations. No refit on train+val. Test is never read.",
        "models": [],
    }
    score_rows = []
    for best in selected:
        family = best["model"]
        path = model_dir / f"{family}.joblib"
        temporary = path.with_name(path.name + ".tmp")
        joblib.dump(best["pipeline"], temporary, compress=3)
        reloaded = joblib.load(temporary)
        if not np.array_equal(reloaded.predict(x_val), best["predictions"]):
            raise RuntimeError(f"Reload changed validation predictions for {family}")
        temporary.replace(path)
        metadata = {
            "model": family, "artifact": str(path.resolve()), "sha256": file_hash(path),
            "params": best["params"], "selected_trial": best["selected_trial"],
            "configurations_tried": best["configurations_tried"],
            "validation_metrics": best["metrics"], "reload_verified": True,
            "vocabulary_size": len(reloaded.named_steps["vectorizer"].vocabulary_) if family != "dummy" else 0,
        }
        summary["models"].append(metadata)
        score_rows.append({
            "model": family, **best["metrics"], "selected_trial": best["selected_trial"],
            "params": json.dumps(best["params"], sort_keys=True),
        })
        columns = [column for column in ("id", "post_id", "text", text_column) if column in val]
        predictions = val[list(dict.fromkeys(columns))].copy()
        predictions["label"] = y_val
        predictions["prediction"] = best["predictions"]
        predictions["outcome"] = [
            ("TP" if actual == 1 else "TN") if actual == predicted
            else ("FN" if actual == 1 else "FP")
            for actual, predicted in zip(y_val, best["predictions"])
        ]
        write_csv(report_dir / f"val_predictions_{family}.csv", predictions)
    write_csv(report_dir / "validation_trials.csv", pd.DataFrame(trials))
    write_csv(report_dir / "validation_results.csv", pd.DataFrame(score_rows))
    write_json(report_dir / "training_summary.json", summary)
    write_json(model_dir / "manifest.json", summary)
    lines = [
        "# Kết quả ML baseline trên validation", "",
        f"- Chế độ: {summary['mode']}; seed: {seed}.",
        f"- Train: {len(train)} câu; validation: {len(val)} câu.",
        f"- Đầu vào: `{text_column}`; nhãn 0 = non-toxic, 1 = toxic.",
        "- Chọn Macro-F1 cao nhất; hòa điểm xét F1-toxic, Recall-toxic, rồi cấu hình đầu tiên.",
        "- Chỉ fit trên train; chưa đánh giá test và không refit trên train+val.", "",
        "| Mô hình | Accuracy | Precision toxic | Recall toxic | F1 toxic | Macro-F1 | TN / FP / FN / TP |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in score_rows:
        values = " | ".join(f"{row[key]:.4f}" for key in ("accuracy", "precision_toxic", "recall_toxic", "f1_toxic", "macro_f1"))
        counts = " / ".join(str(row[key]) for key in ("tn", "fp", "fn", "tp"))
        lines.append(f"| {row['model']} | {values} | {counts} |")
    lines.extend(["", "## Tham số được chọn", ""])
    for best in selected:
        lines.append(f"- `{best['model']}`: `{json.dumps(best['params'], sort_keys=True)}`")
    lines.extend([
        "", "Các điểm trên được dùng để chọn mô hình, không phải kết quả đánh giá độc lập cuối cùng.",
        "Xem `validation_trials.csv` cho toàn bộ cấu hình và `val_predictions_*.csv` để phân tích lỗi.",
        f"Baseline tốt nhất theo tiêu chí đã chốt: `{best_baseline}`.",
        f"Manifest ghi hash đầu vào, phiên bản thư viện và kiểm tra lưu/tải lại toàn bộ {len(val)} dự đoán validation.", "",
    ])
    (report_dir / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--val", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--text-column", default="text_tokenized")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="One configuration per family; separate default output directories")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        defaults = training_defaults() if any(v is None for v in (args.train, args.val, args.model_dir, args.report_dir)) else {}
        train_path, val_path = args.train or defaults['train'], args.val or defaults['val']
        model_dir = args.model_dir or defaults['baseline_models']
        report_dir = args.report_dir or defaults['baseline_reports']
        if args.quick:
            if args.model_dir is None:
                model_dir /= 'quick'
            if args.report_dir is None:
                report_dir /= 'quick'
        run_training(
            train_path, val_path, model_dir, report_dir,
            text_column=args.text_column, seed=args.seed, quick=args.quick,
        )
    except (OSError, ValueError, RuntimeError, KeyError, pd.errors.ParserError) as exc:
        LOG.error("Training failed: %s", exc)
        return 1
    LOG.info("Saved models to %s; reports to %s", model_dir, report_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
