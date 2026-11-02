"""Checks for character features, grouped diagnostics, and original preservation."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd

from scripts import train_baselines as base
from scripts import train_improved_baselines as improved


class ImprovedBaselineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.train_path = self.root / "train.csv"
        self.val_path = self.root / "val.csv"
        self.train = self.make_frame("train", 12)
        self.val = self.make_frame("val", 4)
        self.train.to_csv(self.train_path, index=False, encoding="utf-8-sig")
        self.val.to_csv(self.val_path, index=False, encoding="utf-8-sig")
        self.model_dir = self.root / "improvements"
        self.report_dir = self.root / "reports"
        self.reference_dir = self.root / "original_models"
        self.reference_manifest = self.reference_dir / "manifest.json"

    @staticmethod
    def make_frame(prefix, size):
        return pd.DataFrame([
            {
                "id": f"{prefix}{i}", "post_id": f"{prefix}_post{i // 2}",
                "text_clean": ("đồ ngốc xúc phạm" if i % 2 else "học phí bình thường") + f" {prefix}only{i}",
                "text_tokenized": ("đồ_ngốc xúc_phạm" if i % 2 else "học_phí bình_thường") + f" {prefix}only{i}",
                "label": i % 2, "note": "secret_note label explanation",
            }
            for i in range(size)
        ])

    def create_reference(self):
        base.run_training(
            self.train_path, self.val_path, self.reference_dir,
            self.root / "original_reports", quick=True,
        )

    def test_fixed_grid_has_60_configurations(self):
        self.assertEqual([len(improved.parameter_grid(f)) for f in improved.FAMILIES], [30, 15, 15])
        self.assertEqual([len(improved.parameter_grid(f, True)) for f in improved.FAMILIES], [1, 1, 1])

    def test_feature_branches_read_correct_columns_and_drop_notes(self):
        frame = pd.DataFrame({
            "text_clean": ["characterunique abc", "characterunique xyz"],
            "text_tokenized": ["word_unique a", "word_unique b"],
            "note": ["secret_note", "secret_note"],
        })
        model = improved.build_pipeline("word_char_svm").fit(frame, [0, 1])
        features = model.named_steps["features"].named_transformers_
        word = features["word"].vocabulary_
        char = features["char"].vocabulary_
        self.assertIn("word_unique", word)
        self.assertNotIn("characterunique", word)
        self.assertIn("cha", char)
        self.assertNotIn("wor", char)
        self.assertFalse(any("secret" in token for token in word))
        self.assertFalse(any("sec" in token for token in char))

    def test_group_folds_keep_whole_posts_and_cover_train_once(self):
        folds = improved.make_group_folds(self.train, 42)
        indices = []
        for fit, held in folds:
            self.assertFalse(set(self.train.iloc[fit].post_id) & set(self.train.iloc[held].post_id))
            indices.extend(held.tolist())
        self.assertEqual(sorted(indices), list(range(len(self.train))))

    def test_reject_missing_or_empty_clean_text(self):
        for empty in (True, False):
            frame = self.train.copy()
            if empty:
                frame.loc[0, "text_clean"] = "  "
            else:
                frame = frame.drop(columns="text_clean")
            frame.to_csv(self.train_path, index=False)
            with self.assertRaisesRegex(ValueError, "text_clean"):
                improved.load_inputs(self.train_path, self.val_path)

    def test_reject_reference_from_different_dataset(self):
        self.create_reference()
        with self.train_path.open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "hash differs"):
            improved.load_reference(self.reference_manifest, self.train_path, self.val_path)

    def test_preserve_original_models_and_reports_and_portable_new_artifacts(self):
        self.create_reference()
        protected = [self.train_path, self.val_path, *self.reference_dir.iterdir(), *(self.root / "original_reports").iterdir()]
        original_hashes = {path: base.file_hash(path) for path in protected}
        with patch.object(base.pd, "read_csv", wraps=pd.read_csv) as read:
            summary = improved.run_experiments(
                self.train_path, self.val_path, self.reference_manifest,
                self.model_dir, self.report_dir, quick=True,
            )
            self.assertEqual([call.args[0] for call in read.call_args_list], [self.train_path, self.val_path])
        self.assertTrue(all(base.file_hash(path) == digest for path, digest in original_hashes.items()))
        self.assertEqual(len(summary["models"]), 3)
        self.assertEqual(len(pd.read_csv(self.report_dir / "validation_trials.csv")), 3)
        self.assertEqual(len(pd.read_csv(self.report_dir / "validation_comparison.csv")), 8)
        cv = pd.read_csv(self.report_dir / "group_cv_folds.csv")
        self.assertEqual(len(cv), 12)
        self.assertTrue(cv.post_overlap.eq(0).all())
        self.assertTrue(cv.status.eq("ok").all())
        for metadata in summary["models"]:
            model = joblib.load(metadata["artifact"])
            self.assertEqual(base.file_hash(Path(metadata["artifact"])), metadata["sha256"])
            predictions = model.predict(self.val[improved.INPUT_COLUMNS])
            saved = pd.read_csv(self.report_dir / f"val_predictions_{metadata['model']}.csv")
            self.assertEqual(predictions.tolist(), saved.prediction.tolist())
            blank = pd.DataFrame({"text_clean": ["", "ΩΩΩ"], "text_tokenized": ["", "ΩΩΩ"]})
            self.assertTrue(set(model.predict(blank)) <= {0, 1})
            features = model.named_steps["features"].named_transformers_
            if "word" in features:
                self.assertFalse(any("valonly" in token for token in features["word"].vocabulary_))
            if "char" in features:
                self.assertNotIn("val", features["char"].vocabulary_)
        # A fresh process outside the repository must load without custom modules.
        path = self.model_dir / "word_char_svm.joblib"
        code = (
            "import json,sys,joblib,pandas as pd; "
            "m=joblib.load(sys.argv[1]); "
            "x=pd.read_csv(sys.argv[2],keep_default_na=False); "
            "print(json.dumps(m.predict(x).tolist()))"
        )
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-c", code, str(path), str(self.val_path)],
            cwd=self.root, text=True, capture_output=True, timeout=60, check=True,
        )
        self.assertEqual(len(json.loads(result.stdout)), len(self.val))

    def test_reject_writing_into_original_model_directory(self):
        with self.assertRaisesRegex(ValueError, "separate model directory"):
            improved.run_experiments(
                self.train_path, self.val_path, self.reference_manifest,
                self.reference_dir, self.report_dir, quick=True,
            )

    def test_cli_missing_input_returns_error(self):
        with self.assertLogs("train_improved_baselines", level="ERROR"):
            result = improved.main(["--train", str(self.root / "missing.csv")])
        self.assertEqual(result, 1)


if __name__ == "__main__":
    unittest.main()
