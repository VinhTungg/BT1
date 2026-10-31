# -*- coding: utf-8 -*-
"""Unit tests cho scripts/make_splits.py — dữ liệu tổng hợp có kiểm soát nhóm.

Chạy: python -m unittest discover -s tests -v
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts import make_splits


def write_cleaned_csv(
    path: Path, posts: dict[str, int], toxic_total: int = 0, toxic_every: int = 0
) -> None:
    """Ghi cleaned.csv tổng hợp: mỗi post có ``n`` câu và id duy nhất.

    ``posts``: mapping post_id -> số câu; mỗi câu có text riêng nên không
    sinh duplicate ngoài ý muốn. Cách gán toxic:
      * ``toxic_total`` > 0: số câu đầu tiên (theo id tăng dần) được gán 1,
        phần còn lại 0 (chuỗi toxic dồn về nhóm đầu — dùng để kiểm robust).
      * ``toxic_every`` > 0: mỗi ``toxic_every`` câu có 1 câu toxic,
        phân bố đều các nhóm (dùng để kiểm cân bằng nhãn giữa các tập).
    "    """
    header = "id,post_id,text,text_clean,text_tokenized,label,source_file,source_row\n"
    rows = []
    counter = 0
    for post_id, n in posts.items():
        for _ in range(n):
            if toxic_every > 0:
                label = "1" if counter % toxic_every == 0 else "0"
            else:
                label = "1" if counter < toxic_total else "0"
            rows.append(
                f"id{counter:04d},{post_id},Van ban so {counter},"
                f"Van ban so {counter},Van_ban_so_{counter},{label},src.csv,{counter + 1}"
            )
            counter += 1
    path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")


def substitute_sentence(path: Path, old_index: int, new_text: str) -> None:
    """Thay toàn bộ (text, text_clean, text_tokenized) của một câu theo id."""
    raw = path.read_text(encoding="utf-8")
    old_text = f"Van ban so {old_index}"
    raw = raw.replace(
        f"{old_text},{old_text},Van_ban_so_{old_index}",
        f"{new_text},{new_text},{new_text.replace(' ', '_')}",
    )
    path.write_text(raw, encoding="utf-8")


class SplitTestBase(unittest.TestCase):
    """Khởi tạo thư mục tạm cho input/output của từng test."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.input_csv = self.tmp / "cleaned.csv"
        self.output_dir = self.tmp / "splits"
        self.report_dir = self.tmp / "reports"
        self.addCleanup(self._tmp.cleanup)

    def run_split(self, seed: int = 42, n_candidates: int = 60):
        return make_splits.run_split_pipeline(
            self.input_csv, self.output_dir, self.report_dir,
            seed=seed, n_candidates=n_candidates,
        )


class TestGroupConstraint(SplitTestBase):
    def test_all_sentences_of_same_post_stay_together(self):
        write_cleaned_csv(self.input_csv, {"p1": 100, "p2": 50, "p3": 60, "p4": 40}, 40)
        splits, _ = self.run_split()
        post_sets = {name: set(s["post_id"]) for name, s in splits.items()}
        self.assertFalse(post_sets["train"] & post_sets["val"])
        self.assertFalse(post_sets["train"] & post_sets["test"])
        self.assertFalse(post_sets["val"] & post_sets["test"])
        all_posts = set().union(*post_sets.values())
        self.assertEqual(all_posts, {"p1", "p2", "p3", "p4"})

    def test_no_record_lost_or_added(self):
        write_cleaned_csv(self.input_csv, {"p1": 8, "p2": 9, "p3": 10, "p4": 11}, 5)
        df = pd.read_csv(self.input_csv, encoding="utf-8", dtype=str, keep_default_na=False)
        splits, _ = self.run_split()
        joined = pd.concat(splits.values(), ignore_index=True)
        self.assertEqual(len(joined), len(df))
        self.assertEqual(sorted(joined["id"]), sorted(df["id"]))


class TestRatiosAndDistribution(SplitTestBase):
    def test_ratios_near_target_for_many_groups(self):
        posts = {f"p{i}": 10 for i in range(30)}
        write_cleaned_csv(self.input_csv, posts, 30)
        splits, _ = self.run_split()
        self.assertGreaterEqual(len(splits["train"]) / 300, 0.66)
        self.assertLessEqual(len(splits["train"]) / 300, 0.74)
        self.assertGreaterEqual(len(splits["val"]) / 300, 0.10)
        self.assertLessEqual(len(splits["val"]) / 300, 0.20)
        self.assertGreaterEqual(len(splits["test"]) / 300, 0.10)
        self.assertLessEqual(len(splits["test"]) / 300, 0.20)

    def test_both_classes_present_when_dataset_has_both(self):
        posts = {f"p{i}": 10 + i for i in range(20)}
        write_cleaned_csv(self.input_csv, posts, 40)
        splits, _ = self.run_split()
        for name, s in splits.items():
            self.assertGreater((s["label"] == 1).sum(), 0, name)
            self.assertGreater((s["label"] == 0).sum(), 0, name)

    def test_label_distribution_close_to_overall(self):
        posts = {f"p{i}": 20 for i in range(25)}
        write_cleaned_csv(self.input_csv, posts, toxic_every=5)  # 20% toxic đều
        splits, _ = self.run_split()
        for s in splits.values():
            ratio = (s["label"] == 1).mean()
            self.assertLessEqual(abs(ratio - 0.20), 0.08)


class TestRobustCases(SplitTestBase):
    def test_group_size_highly_uneven(self):
        write_cleaned_csv(self.input_csv, {"big": 50, "p2": 2, "p3": 3, "p4": 4}, 10)
        splits, _ = self.run_split()
        self.assertEqual(sum(len(s) for s in splits.values()), 59)

    def test_missing_post_id_raises(self):
        write_cleaned_csv(self.input_csv, {"": 5, "p2": 10}, 0)
        with self.assertRaises(RuntimeError):
            self.run_split()

    def test_too_few_groups_raises(self):
        write_cleaned_csv(self.input_csv, {"p1": 9, "p2": 5}, 0)
        with self.assertRaises(RuntimeError):
            self.run_split()

    def test_single_class_dataset_still_splits(self):
        # Toàn bộ câu đều toxic: phải chia bình thường, không ép thêm lớp
        posts = {f"p{i}": 8 for i in range(12)}
        write_cleaned_csv(self.input_csv, posts, len(posts) * 8)
        splits, _ = self.run_split()
        self.assertEqual(sum(len(s) for s in splits.values()), 96)
        self.assertEqual(sum((s["label"] == 1).sum() for s in splits.values()), 96)

    def test_bad_input_missing_required_column(self):
        self.input_csv.write_text("id,post_id,text,label\na1,p1,Hi,1\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.run_split()


class TestDeterminism(SplitTestBase):
    def test_same_seed_same_split(self):
        write_cleaned_csv(self.input_csv, {f"p{i}": 5 + i for i in range(15)}, 40)
        first, _ = self.run_split(seed=7)
        second, _ = self.run_split(seed=7)
        for name in ("train", "val", "test"):
            pd.testing.assert_frame_equal(first[name], second[name])

    def test_different_seed_still_produces_valid_split(self):
        write_cleaned_csv(self.input_csv, {f"p{i}": 5 + i for i in range(15)}, 40)
        splits, checks = self.run_split(seed=99)
        self.assertTrue(all(checks["group_leakage"].values()))


class TestDuplicateTextAcrossSplits(SplitTestBase):
    def test_shared_text_between_splits_detected(self):
        shared_text = "LIKE shared comment"
        left = pd.DataFrame({
            "id": ["a1"], "text_clean": [shared_text], "label": [0], "post_id": ["p1"],
        })
        right = pd.DataFrame({
            "id": ["b9"], "text_clean": [shared_text], "label": [1], "post_id": ["p2"],
        })
        reports = make_splits.find_cross_split_text_duplicates(
            {"train": left, "val": right, "test": pd.DataFrame({"id": [], "text_clean": []})}
        )
        self.assertTrue(any(r["pair"] == "train/val" for r in reports))

    def test_no_shared_text_no_report(self):
        left = pd.DataFrame({"id": ["a1"], "text_clean": ["mot"], "label": [0], "post_id": ["p1"]})
        right = pd.DataFrame({"id": ["b9"], "text_clean": ["hai"], "label": [1], "post_id": ["p2"]})
        reports = make_splits.find_cross_split_text_duplicates(
            {"train": left, "val": right, "test": pd.DataFrame({"id": [], "text_clean": []})}
        )
        self.assertEqual(reports, [])

    def test_duplicate_in_real_pipeline_flow_reported_or_clean(self):
        """Smoke test toàn bộ hàm split với một bảng có 2 câu trùng text."""
        write_cleaned_csv(self.input_csv, {"p1": 10, "p2": 10, "p3": 10}, 5)
        substitute_sentence(self.input_csv, 0, "Chua cham thuc")
        substitute_sentence(self.input_csv, 15, "Chua cham thuc")
        splits, stats = self.run_split()
        self.assertIn("cross_split_text_duplicates", stats)
        self.assertIn("group_leakage", stats)
        self.assertEqual(sum(len(s) for s in splits.values()), 30)


if __name__ == "__main__":
    unittest.main()
