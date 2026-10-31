# -*- coding: utf-8 -*-
"""Unit tests cho scripts/audit_data.py — dùng dữ liệu tổng hợp.

Chạy: python -m unittest discover -s tests -v
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts import audit_data

HEADER = "id,post_id,text,label,source_url,collected_at,extraction_method,note\n"


class AuditTestBase(unittest.TestCase):
    """Helper: khởi tạo thư mục input/output tạm cho từng test."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.input_dir = self.tmp / "annotations"
        self.input_dir.mkdir()
        self.output_csv = self.tmp / "merged.csv"
        self.report_dir = self.tmp / "reports"
        self.addCleanup(self._tmp.cleanup)

    def write_csv(self, name: str, rows: str, encoding: str = "utf-8") -> Path:
        """Ghi file CSV tổng hợp (rows = chuỗi các dòng dữ liệu)."""
        path = self.input_dir / name
        path.write_text(HEADER + rows, encoding=encoding)
        return path

    def run_audit(self):
        """Chạy audit với paths tạm."""
        return audit_data.run_audit(self.input_dir, self.output_csv, self.report_dir)


class TestValidRead(AuditTestBase):
    def test_reads_valid_csv_and_adds_traceability(self):
        self.write_csv("mem.csv", "a1,p1,Xin chao,1,u1,t1,m,\n")
        merged, stats = self.run_audit()
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged.iloc[0]["source_file"], "mem.csv")
        self.assertEqual(int(merged.iloc[0]["source_row"]), 1)
        self.assertEqual(stats["valid_records"], 1)

    def test_label_with_whitespace_is_parsed(self):
        self.write_csv("mem.csv", "a1,p1,Text, 0 ,u1,t1,m,\n")
        merged, _ = self.run_audit()
        self.assertEqual(int(merged.iloc[0]["label"]), 0)


class TestInvalidData(AuditTestBase):
    def test_missing_column_reported(self):
        bad = self.input_dir / "bad.csv"
        bad.write_text("id,text,label\na1,Hi,1\n", encoding="utf-8")
        with self.assertRaises(FileNotFoundError):
            self.run_audit()

    def test_missing_column_in_one_file_does_not_lose_other_files(self):
        bad = self.input_dir / "bad.csv"
        bad.write_text("id,text,label\na1,Hi,1\n", encoding="utf-8")
        self.write_csv("good.csv", "a2,p1,Ok,0,u1,t1,m,\n")
        merged, _ = self.run_audit()
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged.iloc[0]["source_file"], "good.csv")

    def test_missing_label_flagged(self):
        self.write_csv("mem.csv", "a1,p1,Text,,u1,t1,m,\n")
        with self.assertRaises(RuntimeError):
            self.run_audit()

    def test_label_outside_01_flagged(self):
        self.write_csv("mem.csv", "a1,p1,Text,2,u1,t1,m,\n")
        with self.assertRaises(RuntimeError):
            self.run_audit()

    def test_whitespace_only_text_flagged(self):
        self.write_csv("mem.csv", "a1,p1,   ,0,u1,t1,m,\n")
        with self.assertRaises(RuntimeError):
            self.run_audit()

    def test_empty_post_id_flagged(self):
        self.write_csv("mem.csv", "a1,,Text,0,u1,t1,m,\n")
        with self.assertRaises(RuntimeError):
            self.run_audit()

    def test_empty_file_reported_not_crash(self):
        (self.input_dir / "empty.csv").write_text("", encoding="utf-8")
        self.write_csv("good.csv", "a1,p1,Ok,0,u1,t1,m,\n")
        merged, stats = self.run_audit()
        self.assertEqual(len(merged), 1)
        self.assertTrue(any(e["file"] == "empty.csv" for e in stats["parse_errors"]))

    def test_encoding_error_reported_not_crash(self):
        (self.input_dir / "bad_bytes.csv").write_bytes(b"\xff\xfe\xff\xff,ab,cd\n")
        self.write_csv("good.csv", "a1,p1,Ok,0,u1,t1,m,\n")
        merged, stats = self.run_audit()
        self.assertEqual(len(merged), 1)
        self.assertTrue(any(e["file"] == "bad_bytes.csv" for e in stats["parse_errors"]))

    def test_no_nan_like_strings_in_output(self):
        self.write_csv("mem.csv", "a1,p1,Text,1,u1,t1,,note here\n")
        merged, _ = self.run_audit()
        merged = merged.fillna("<missing>")
        self.assertNotIn("nan", merged["note"].astype(str).str.lower())
        self.assertNotIn("None", merged["note"].astype(str))


class TestDuplicates(AuditTestBase):
    def test_duplicate_id_within_file_removed_with_log(self):
        self.write_csv("mem.csv", "a1,p1,Text A,0,u1,t1,m,\na1,p1,Text A,0,u1,t1,m,\n")
        merged, stats = self.run_audit()
        self.assertEqual(len(merged), 1)
        self.assertEqual(stats["removed_duplicates"], 1)
        issues = pd.read_csv(self.report_dir / "data_issues.csv", keep_default_na=False)
        self.assertIn("duplicate_id", issues["issue"].tolist())

    def test_duplicate_id_across_files_keeps_first(self):
        self.write_csv("b_file.csv", "a1,p1,Text,1,u1,t1,m,\n")
        self.write_csv("a_file.csv", "a1,p1,Text,1,u1,t1,m,\n")
        merged, stats = self.run_audit()
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged.iloc[0]["source_file"], "a_file.csv")
        self.assertEqual(stats["removed_duplicates"], 1)

    def test_duplicate_text_different_id_reported_not_removed(self):
        self.write_csv("mem.csv", "a1,p1,Same text,1,u1,t1,m,\na2,p1,Same text,1,u1,t1,m,\n")
        merged, stats = self.run_audit()
        self.assertEqual(len(merged), 2)  # không tự loại khi chưa rõ nguyên nhân
        self.assertEqual(stats["duplicate_text_records"], 2)
        issues = pd.read_csv(self.report_dir / "data_issues.csv", keep_default_na=False)
        self.assertIn("duplicate_text_different_id", issues["issue"].tolist())

    def test_conflicting_labels_reported_not_silently_resolved(self):
        self.write_csv("mem.csv", "a1,p1,Same,1,u1,t1,m,\na2,p1,Same,0,u1,t1,m,\n")
        merged, stats = self.run_audit()
        self.assertEqual(len(merged), 2)
        self.assertEqual(stats["conflicting_label_records"], 2)
        labels = sorted(merged["label"].tolist())
        self.assertEqual(labels, [0, 1])  # không majority vote, không đổi nhãn
        issues = pd.read_csv(self.report_dir / "data_issues.csv", keep_default_na=False)
        self.assertIn("conflicting_label_same_text", issues["issue"].tolist())

    def test_multiple_sentences_same_post_kept(self):
        self.write_csv("mem.csv", "a1,p1,One,0,u1,t1,m,\na2,p1,Two,1,u1,t1,m,\n")
        merged, stats = self.run_audit()
        self.assertEqual(len(merged), 2)
        self.assertEqual(stats["unique_post_ids"], 1)


class TestOutputsAndSafety(AuditTestBase):
    def test_does_not_overwrite_annotation_files(self):
        path = self.write_csv("mem.csv", "a1,p1,Text,0,u1,t1,m,\n")
        before = path.read_bytes()
        self.run_audit()
        self.assertEqual(path.read_bytes(), before)

    def test_output_files_created(self):
        self.write_csv("mem.csv", "a1,p1,Text,0,u1,t1,m,\n")
        self.run_audit()
        self.assertTrue(self.output_csv.exists())
        for f in ("data_audit.md", "data_audit.json", "data_issues.csv"):
            self.assertTrue((self.report_dir / f).exists(), f)

    def test_record_count_conservation_reported(self):
        self.write_csv("mem.csv", "a1,p1,Text,0,u1,t1,m,\n")
        _, stats = self.run_audit()
        self.assertTrue(stats["record_count_check"])

    def test_preserves_leading_zero_ids(self):
        self.write_csv("mem.csv", "00ab,p1,Text,0,u1,t1,m,\n")
        merged, _ = self.run_audit()
        self.assertEqual(merged.iloc[0]["id"], "00ab")


if __name__ == "__main__":
    unittest.main()
