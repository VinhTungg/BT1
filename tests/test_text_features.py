"""Behavior checks for meaning preservation and train-only vocabularies."""
import unittest

import pandas as pd

from src.text_features import CommentPreprocessor, build_features, normalize_for_features


class TextFeatureTests(unittest.TestCase):
    def test_normalization_keeps_negation_and_ambiguous_units(self):
        self.assertEqual(normalize_for_features("ko thik 100k và 2k5, k, đk", "slang"),
                         "không thích 100k và 2k5, k, đk")
        self.assertEqual(normalize_for_features("KO ko", "slang"), "KO không")

    def test_normalization_is_idempotent_and_preserves_toxic_signals(self):
        text = "[MENTION] ko được chửi người khác!!! 👩🏽‍💻 https://example.com"
        for profile in ("preserve", "slang"):
            normalized = normalize_for_features(text, profile)
            self.assertEqual(normalize_for_features(normalized, profile), normalized)
            self.assertIn("chửi người khác!!! 👩🏽‍💻", normalized)

    def test_pipeline_keeps_placeholders_emoji_and_handles_empty_text(self):
        frame = CommentPreprocessor().fit_transform(["[MENTION] hay 👨‍👩‍👧‍👦", "", "x"])
        self.assertIn("[MENTION]", frame.iloc[0].text_tokenized.split())
        self.assertIn("👨‍👩‍👧‍👦", frame.iloc[0].text_tokenized.split())
        self.assertEqual(frame.iloc[1].text_tokenized, "")
        with self.assertRaises(TypeError):
            CommentPreprocessor().transform(pd.DataFrame({"note": ["secret"]}))

    def test_no_validation_vocabulary_and_no_metadata_features(self):
        prep = CommentPreprocessor()
        train = prep.fit_transform(["học phí a 😂", "đồ ngốc b 😡"])
        train["note"] = "secret_note"
        held = prep.transform(["validationonlyword"])
        for kind in ("bow", "tfidf", "char", "word_char"):
            features = build_features(kind).fit(train)
            before = features.get_feature_names_out().tolist()
            self.assertFalse(any("secret" in token for token in before))
            features.transform(held)
            self.assertEqual(before, features.get_feature_names_out().tolist())
            self.assertFalse(any("validationonly" in token for token in before))


if __name__ == "__main__":
    unittest.main()
