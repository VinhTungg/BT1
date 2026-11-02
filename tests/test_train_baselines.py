"""Behavior checks for training, holdout isolation, and portable artifacts."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd

from scripts import train_baselines as baseline


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.train_path = self.root / "train.csv"
        self.val_path = self.root / "val.csv"
        self.models = self.root / "models"
        self.reports = self.root / "reports"
        self.train = self.make_rows("train", 8)
        self.val = self.make_rows("val", 4)
        self.train.to_csv(self.train_path, index=False, encoding="utf-8-sig")
        self.val.to_csv(self.val_path, index=False, encoding="utf-8-sig")

    @staticmethod
    def make_rows(prefix, count):
        return pd.DataFrame([
            {
                "id": f"{prefix}{i}", "post_id": f"{prefix}_post{i % 2}",
                "text_tokenized": ("đồ_ngốc xúc_phạm" if i % 2 else "học_phí bình_thường") + f" {prefix}only{i}",
                "label": i % 2, "note": "secret_note label explanation",
            }
            for i in range(count)
        ])

    def test_full_grid_is_72_candidates_and_quick_is_four(self):
        self.assertEqual(sum(len(baseline.parameter_grid(f)) for f in baseline.FAMILIES), 72)
        self.assertEqual(sum(len(baseline.parameter_grid(f, True)) for f in baseline.FAMILIES), 4)

    def test_reject_invalid_labels_empty_input_and_single_class(self):
        for change in ("invalid_label", "empty_text", "one_class", "duplicate_id"):
            with self.subTest(change=change):
                frame = self.train.copy()
                if change == "invalid_label":
                    frame["label"] = "2"
                elif change == "empty_text":
                    frame.loc[0, "text_tokenized"] = "  "
                elif change == "one_class":
                    frame["label"] = "0"
                else:
                    frame.loc[0, "id"] = frame.loc[1, "id"]
                frame.to_csv(self.train_path, index=False)
                with self.assertRaises(ValueError):
                    baseline.load_split(self.train_path)

    def test_reject_leakage_by_post_id_id_or_text(self):
        for column in ("post_id", "id", "text_tokenized"):
            with self.subTest(column=column):
                val = self.val.copy()
                val.loc[0, column] = self.train.loc[0, column]
                with self.assertRaisesRegex(ValueError, "leakage"):
                    baseline.validate_separation(self.train, val, "text_tokenized")

    def test_vectorizer_keeps_segmented_words_emoji_and_single_letters(self):
        pipeline = baseline.build_pipeline("bow_nb")
        pipeline.fit(["Học_phí a 😂", "đồ_ngốc b 😡"], [0, 1])
        vocabulary = pipeline.named_steps["vectorizer"].vocabulary_
        self.assertTrue({"học_phí", "a", "😂"} <= set(vocabulary))

    def test_majority_prediction_cannot_hide_zero_toxic_recall(self):
        metrics = baseline.evaluate([0, 0, 0, 1], [0, 0, 0, 0])
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["recall_toxic"], 0)
        self.assertEqual(metrics["f1_toxic"], 0)
        self.assertAlmostEqual(metrics["macro_f1"], 3 / 7)
        self.assertEqual((metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]), (3, 0, 1, 0))

    def test_end_to_end_artifacts_fit_only_train_and_do_not_use_notes(self):
        before = {p: p.read_bytes() for p in (self.train_path, self.val_path)}
        with patch.object(baseline.pd, "read_csv", wraps=pd.read_csv) as read:
            summary = baseline.run_training(
                self.train_path, self.val_path, self.models, self.reports, quick=True,
            )
            self.assertEqual([call.args[0] for call in read.call_args_list], [self.train_path, self.val_path])
        self.assertEqual(len(summary["models"]), 5)
        self.assertEqual(len(pd.read_csv(self.reports / "validation_trials.csv")), 5)
        for metadata in summary["models"]:
            pipeline = joblib.load(metadata["artifact"])
            predictions = pipeline.predict(self.val["text_tokenized"].tolist())
            self.assertEqual(len(predictions), len(self.val))
            self.assertTrue(set(predictions) <= {0, 1})
            self.assertTrue(metadata["reload_verified"])
            if metadata["model"] != "dummy":
                vocabulary = pipeline.named_steps["vectorizer"].vocabulary_
                self.assertFalse(any("valonly" in token or "secret_note" in token for token in vocabulary))
                # Empty/OOV input must produce a valid prediction, not crash.
                self.assertTrue(set(pipeline.predict(["", "unseen_word"])) <= {0, 1})
            saved = pd.read_csv(self.reports / f"val_predictions_{metadata['model']}.csv")
            self.assertEqual(saved["prediction"].tolist(), predictions.tolist())
        manifest = json.loads((self.models / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["train"]["sha256"], baseline.file_hash(self.train_path))
        self.assertTrue(all(path.read_bytes() == content for path, content in before.items()))

    def test_cli_missing_input_returns_failure_without_publishing_models(self):
        with self.assertLogs("train_baselines", level="ERROR"):
            result = baseline.main([
                "--train", str(self.root / "missing.csv"), "--val", str(self.val_path),
                "--model-dir", str(self.models), "--report-dir", str(self.reports),
            ])
        self.assertEqual(result, 1)
        self.assertFalse(self.models.exists())


if __name__ == "__main__":
    unittest.main()
