# -*- coding: utf-8 -*-
"""Integration test chạy toàn bộ pipeline trên DỮ LIỆU THẬT của repository.

Test không thay thế output thật; mọi thứ ghi vào thư mục tạm, file annotation
gốc được xác nhận không bị sửa. Bỏ qua nếu annotations/ không có dữ liệu.

Chạy: python -m unittest discover -s tests -v
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts import audit_data, make_splits
from src import preprocessing

REPO_ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS_DIR = REPO_ROOT / "annotations"
EXPECTED_TOTAL = 2314  # từ manifest.json của nhóm


@unittest.skipUnless(
    ANNOTATIONS_DIR.exists() and list(ANNOTATIONS_DIR.glob("*.csv")),
    "Không có dữ liệu annotation thật trong repository.",
)
class TestRealDataPipeline(unittest.TestCase):
    """Pipeline đầy đủ audit → preprocess → split trên dữ liệu thật."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        # Copy annotation thật vào thư mục tạm để đảm bảo không bao giờ
        # ghi đè lên dữ liệu gốc trong quá trình test.
        input_dir = self.tmp / "annotations"
        input_dir.mkdir()
        for csv in ANNOTATIONS_DIR.glob("*.csv"):
            shutil.copy2(csv, input_dir / csv.name)
        self.input_dir = input_dir
        self.output_csv = self.tmp / "data" / "processed" / "merged.csv"
        self.report_dir = self.tmp / "reports"
        self.splits_dir = self.tmp / "data" / "splits"
        self.addCleanup(self._tmp.cleanup)

    def _run_audit(self):
        return audit_data.run_audit(self.input_dir, self.output_csv, self.report_dir)

    def _run_preprocess(self):
        cleaned_csv = self.tmp / "data" / "processed" / "cleaned.csv"
        preprocessing.run_preprocessing(self.output_csv, cleaned_csv)
        return cleaned_csv

    def _run_split(self, cleaned_csv: Path, seed: int = 42):
        return make_splits.run_split_pipeline(
            cleaned_csv, self.splits_dir, self.report_dir, seed=seed
        )

    def test_full_pipeline_on_real_data(self):
        # --- Bước 1: Audit & merge ---
        merged, stats = self._run_audit()
        self.assertEqual(stats["valid_records"], EXPECTED_TOTAL)
        self.assertTrue(stats["record_count_check"])
        self.assertEqual(sorted(merged["label"].unique()), [0, 1])
        self.assertEqual(merged["id"].nunique(), len(merged))
        self.assertGreaterEqual(merged["post_id"].nunique(), 3)

        # File annotation gốc không bị sửa
        for csv in ANNOTATIONS_DIR.glob("*.csv"):
            self.assertEqual(
                csv.read_bytes(), (self.input_dir / csv.name).read_bytes(), csv.name
            )

        # --- Bước 2: Preprocessing ---
        cleaned_csv = self._run_preprocess()
        cleaned = pd.read_csv(cleaned_csv, encoding="utf-8", dtype=str, keep_default_na=False)
        self.assertEqual(len(cleaned), EXPECTED_TOTAL)
        self.assertEqual(cleaned["label"].tolist(), merged["label"].astype(str).tolist())
        self.assertEqual(cleaned["post_id"].tolist(), merged["post_id"].tolist())
        self.assertFalse((cleaned["text_clean"].str.strip() == "").any())
        # Idempotency trên toàn bộ nền text thật
        re_cleaned = [preprocessing.clean_text(t) for t in cleaned["text_clean"]]
        self.assertEqual(re_cleaned, cleaned["text_clean"].tolist())

        # --- Bước 3: Group-aware split ---
        splits, split_stats = self._run_split(cleaned_csv)
        self.assertTrue(all(split_stats["group_leakage"].values()))
        self.assertTrue(all(split_stats["record_checks"].values()))
        n_samples = sum(len(s) for s in splits.values())
        self.assertEqual(n_samples, EXPECTED_TOTAL)
        self.assertEqual(
            len(set().union(*(set(s["post_id"]) for s in splits.values()))),
            merged["post_id"].nunique(),
        )
        # Cả ba tập có ghi để huấn luyện
        for name in ("train", "val", "test"):
            self.assertGreater(len(splits[name]), 0, name)

        # --- Bước 4: Reproducibility với cùng seed ---
        splits_again, _ = self._run_split(cleaned_csv, seed=42)
        for name in ("train", "val", "test"):
            pd.testing.assert_frame_equal(splits[name], splits_again[name])


if __name__ == "__main__":
    unittest.main()
