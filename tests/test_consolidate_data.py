"""Data-loss and leakage guards use synthetic fixtures only."""
import unittest
from scripts.consolidate_data import validate


def row(identifier, text, label='0', post='group:1'):
    return dict(id=identifier, text=text, text_clean=text, text_tokenized=text,
                label=label, post_id=post)


class ConsolidationTests(unittest.TestCase):
    def test_short_toxic_and_emoji_are_retained(self):
        rows = [row('001', 'Ngu!', '1'), row('002', '😀')]
        self.assertEqual(validate({'train': rows})['train']['samples'], 2)
        self.assertEqual(rows[0]['id'], '001')

    def test_duplicate_with_conflicting_labels_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Conflicting labels'):
            validate({'train': [row('a', 'Hello', '0'), row('b', 'HELLO', '1')]})

    def test_post_alias_leakage_is_rejected(self):
        rows = {'train': [row('a', 'A', '0'), row('b', 'B', '1')],
                'test': [row('c', 'C', '0', 'other:1'), row('d', 'D', '1', 'other:1')]}
        with self.assertRaisesRegex(ValueError, 'Post leakage'):
            validate(rows)

    def test_empty_and_bad_label_are_rejected(self):
        for bad in (row('a', ''), row('a', 'hello', '2')):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate({'train': [bad]})


if __name__ == '__main__':
    unittest.main()
