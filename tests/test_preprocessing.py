# -*- coding: utf-8 -*-
"""Unit tests cho src/preprocessing.py — dữ liệu tổng hợp và ví dụ underthesea thực tế.

Chạy: python -m unittest discover -s tests -v
"""

import unittest

import pandas as pd

from src import preprocessing


class TestUnicodeNfc(unittest.TestCase):
    def test_nfd_input_converted_to_nfc(self):
        decomposed = "Sin\u0068 vie\u0302\u0303n \u0111ang ho\u0323c"  # dấu phụ tách rời
        nfc = preprocessing.normalize_unicode_nfc(decomposed)
        self.assertEqual(nfc, "Sinh viễn đang học")
        # NFC phải idempotent
        self.assertEqual(preprocessing.normalize_unicode_nfc(nfc), nfc)

    def test_equivalent_forms_normalize_identically(self):
        # Chuỗi có dấu precomposed (U+1EC5) và dạng decomposed (e + U+0302 + U+0303)
        # sau NFC phải trùng nhau hoàn toàn.
        composed = "Sinh vi\u1ec5n \u0111ang ho\u0323c"
        decomposed = "Sinh vie\u0302\u0303n \u0111ang ho\u0323c"
        nfc = preprocessing.normalize_unicode_nfc(composed)
        self.assertEqual(
            preprocessing.normalize_unicode_nfc(decomposed), nfc
        )
        self.assertEqual(preprocessing.normalize_unicode_nfc(nfc), nfc)

    def test_diacritics_preserved(self):
        text = "Sinh viên đang học: ế ề ệ ả ạ đ"
        cleaned = preprocessing.clean_text(text)
        for ch in ("ế", "ệ", "ả", "đ"):
            self.assertIn(ch, cleaned)

    def test_diacritics_not_stripped(self):
        text = "Bảo vệ nhà xe phạt tiền sai quy định"
        cleaned = preprocessing.clean_text(text)
        self.assertEqual(cleaned, text)  # không mất dấu, không lower-case


class TestWhitespace(unittest.TestCase):
    def test_collapses_multiple_spaces_tabs_newlines(self):
        self.assertEqual(
            preprocessing.clean_text("  Sinh   viên \t đang \n học  "),
            "Sinh viên đang học",
        )

    def test_special_unicode_spaces_normalized(self):
        self.assertEqual(
            preprocessing.clean_text("Sinh\u00a0\u3000viên"),
            "Sinh viên",
        )


class TestCleaning(unittest.TestCase):
    def test_control_and_zero_width_removed(self):
        text = "Sinh\u200b vie\u061cn\u200d đang học"
        cleaned = preprocessing.clean_text(text)
        self.assertNotIn("\u200b", cleaned)
        self.assertNotIn("\u200d", cleaned)

    def test_punctuation_preserved(self):
        self.assertEqual(preprocessing.clean_text("Câu hỏi: ???"), "Câu hỏi: ???")

    def test_emoji_preserved(self):
        self.assertEqual(preprocessing.clean_text("Hay :)) 😂"), "Hay :)) 😂")

    def test_url_preserved(self):
        url = "https://example.com/a?b=1"
        self.assertIn(url, preprocessing.clean_text(f"Xem {url} nhé"))

    def test_toxic_words_preserved(self):
        text = "Mấy tk lạo này đéo biết đau à loz"
        self.assertEqual(preprocessing.clean_text(text), text)

    def test_negation_words_preserved(self):
        self.assertIn("không", preprocessing.clean_text("không đồng ý"))
        self.assertIn("chẳng", preprocessing.clean_text("chẳng phải"))

    def test_empty_string(self):
        self.assertEqual(preprocessing.clean_text(""), "")
        self.assertEqual(preprocessing.segment_vietnamese(""), "")

    def test_non_string_raises_type_error(self):
        for bad in (None, 123, ["a"], 1.5):
            with self.assertRaises(TypeError):
                preprocessing.clean_text(bad)
            with self.assertRaises(TypeError):
                preprocessing.segment_vietnamese(bad)

    def test_idempotent(self):
        text = "  Sinh vi\u00ean\u200b   đt học \t phí  "
        once = preprocessing.clean_text(text)
        self.assertEqual(preprocessing.clean_text(once), once)


class TestUnderTheseaSegmentation(unittest.TestCase):
    def test_basic_multiword_joins(self):
        result = preprocessing.segment_vietnamese("Sinh viên đang đóng học phí")
        self.assertIn("Sinh_viên", result)
        self.assertIn("học_phí", result)
        self.assertIn("đang", result)

    def test_output_is_single_space_separated(self):
        for text in (
            "Sinh viên đang học Upon a time",
            "Ôi dào t còn bị bảo vệ đấm đây:))",
            "Rất ghét bảo vệ cái nhà xe này!!!!",
        ):
            result = preprocessing.segment_vietnamese(text)
            self.assertEqual(" ".join(result.split()), result)
            self.assertTrue(result.strip())

    def test_does_not_destroy_toxic_content(self):
        result = preprocessing.segment_vietnamese("mấy tk già loz đéo biết đau")
        joined = result.replace("_", "")
        for word in ("loz", "đéo", "tk"):
            self.assertIn(word, joined)

    def test_mention_placeholder_preserved(self):
        result = preprocessing.segment_vietnamese("[MENTION] sao 2 người lại chui ra đây")
        self.assertIn("[", result)
        self.assertIn("]", result)

    def test_original_text_untouched_by_function_design(self):
        text = "Sinh viên đang học"
        pre_token = preprocessing.segment_vietnamese(text)
        self.assertIn("Sinh_viên", pre_token)
        self.assertEqual(text, "Sinh viên đang học")  # tham số gốc không đổi


class TestDataFrameApplied(unittest.TestCase):
    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "id": ["a1", "a2", "a3"],
                "post_id": ["p1", "p1", "p2"],
                "text": ["  Sinh viên   học ", "Không biết 😑", "Đã nói rồi:))"],
                "label": ["0", "1", "0"],
                "note": ["", "x", ""],
            }
        )

    def test_row_count_and_key_columns_unchanged(self):
        out = preprocessing.preprocess_dataframe(self._frame())
        self.assertEqual(len(out), 3)
        self.assertEqual(out["label"].tolist(), ["0", "1", "0"])
        self.assertEqual(out["post_id"].tolist(), ["p1", "p1", "p2"])
        self.assertEqual(out["text"].tolist(), self._frame()["text"].tolist())

    def test_columns_added(self):
        out = preprocessing.preprocess_dataframe(self._frame())
        for col in ("text_clean", "text_tokenized"):
            self.assertIn(col, out.columns)

    def test_no_empty_text_clean(self):
        out = preprocessing.preprocess_dataframe(self._frame())
        self.assertFalse((out["text_clean"].str.strip() == "").any())

    def test_no_nan_like_strings(self):
        out = preprocessing.preprocess_dataframe(self._frame())
        flat = out.astype(str).values.ravel()
        self.assertNotIn("nan", [str(v).lower() for v in flat])

    def test_missing_text_column_raises(self):
        frame = self._frame().drop(columns=["text"])
        with self.assertRaises(KeyError):
            preprocessing.preprocess_dataframe(frame)


if __name__ == "__main__":
    unittest.main()
