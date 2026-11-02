"""Synthetic local fixtures; no Facebook/network requests or actual training."""
import csv
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from collect_comments import FIELDS, normalize_text
from scripts.prepare_expanded_training import (ROOT, TRAIN_FIELDS, load_frozen, main,
                                                read_decisions, select_ready, sha256)


def raw(text, post='2'):
    return dict(id=hashlib.sha256(normalize_text(text).encode()).hexdigest()[:24],
                post_id=f'2k5ptit:{post}', text=text, label='',
                source_url=f'https://www.facebook.com/groups/2k5ptit/posts/{post}/',
                collected_at='2026-11-01T00:00:00Z', extraction_method='comment_aria_label')


def decision(row, label=0):
    return {key: row[key] for key in ('id', 'post_id', 'source_url')} | dict(
        label=label, review_status='needs_context' if label is None else 'ai_semantic_decided',
        reason='Synthetic fixture decision', reviewer='Fixture')


OLD = [dict(id='old', post_id='old_alias:1', text='old message', text_clean='old message',
            text_tokenized='old_message', label='0')]


class ExpandedSelectionTests(unittest.TestCase):
    def select(self, rows, decisions):
        return select_ready(rows, {r['id']: r for r in decisions}, {'2': {}}, OLD, tokenizer=lambda value: value)

    def test_only_decided_unique_meaningful_new_text_is_ready(self):
        rows = [raw('Xin tư vấn'), raw('XIN TƯ VẤN'), raw('Chưa đủ ngữ cảnh'),
                raw('[MENTION] 0123456789'), raw('old message'), raw('Chưa được đọc')]
        ready, excluded, counts = self.select(rows, [decision(r, None if i == 2 else 0) for i, r in enumerate(rows[:5])])
        self.assertEqual([r['id'] for r in ready], [rows[0]['id']])
        self.assertEqual(set(r['status'] for r in excluded),
                         {'duplicate_clean', 'needs_context', 'no_linguistic_content', 'old_text_overlap', 'unreviewed'})
        self.assertEqual(sum(counts.values()), len(rows))

    def test_conflicting_case_variants_are_both_quarantined(self):
        rows = [raw('Nội dung'), raw('NỘI DUNG')]
        ready, excluded, counts = self.select(rows, [decision(rows[0], 0), decision(rows[1], 1)])
        self.assertEqual(ready, [])
        self.assertEqual(counts, {'conflicting_clean_labels': 2})

    def test_empty_anonymous_display_name_is_not_message_content(self):
        rows = [raw('Người tham gia ẩn danh'), raw('Anonymous participant 123'),
                raw('Người tham gia ẩn danh 804 .'),
                raw('Người tham gia ẩn danh 123 cảm ơn bạn')]
        ready, excluded, counts = self.select(rows, [decision(row) for row in rows])
        self.assertEqual([row['id'] for row in ready], [rows[3]['id']])
        self.assertEqual(counts, {'no_linguistic_content': 3, 'ready': 1})

    def test_old_model_input_is_excluded_even_if_clean_text_differs(self):
        row = raw('newly formatted old message')
        ready, excluded, _ = select_ready([row], {row['id']: decision(row)}, {'2': {}}, OLD,
                                          tokenizer=lambda value: 'old_message')
        self.assertEqual(ready, [])
        self.assertEqual(excluded[0]['status'], 'old_model_input_overlap')

    def test_long_same_post_reference_variants_are_deduplicated_or_quarantined(self):
        prefix = 'Chia sẻ tài liệu lập trình và hướng dẫn ôn tập cho các bạn. ' * 2
        rows = [raw(prefix+'[URL]'), raw(prefix+'[MENTION]')]
        ready, _, counts = self.select(rows, [decision(row) for row in rows])
        self.assertEqual([row['id'] for row in ready], [rows[0]['id']])
        self.assertEqual(counts['duplicate_reference_variation'], 1)
        ready, _, counts = self.select(rows, [decision(rows[0], 0), decision(rows[1], 1)])
        self.assertEqual(ready, [])
        self.assertEqual(counts, {'conflicting_reference_labels': 2})
        other_post = raw(prefix+'[MENTION]', post='3')
        ready, _, _ = select_ready([rows[0], other_post],
                                  {row['id']: decision(row) for row in (rows[0], other_post)},
                                  {'2': {}, '3': {}}, OLD, tokenizer=lambda value: value)
        self.assertEqual(len(ready), 2)

    def test_scheme_relative_url_only_is_excluded_but_real_caption_remains(self):
        rows = [raw('//docs.google.com/forms/x?fbclid=tracking'),
                raw('Xin gửi tài liệu //docs.google.com/forms/x?fbclid=tracking'),
                raw('So sánh / A / B')]
        ready, excluded, counts = self.select(rows, [decision(row) for row in rows])
        self.assertEqual([row['id'] for row in ready], [rows[1]['id'], rows[2]['id']])
        self.assertEqual(counts['no_linguistic_content'], 1)
        self.assertNotIn('fbclid', ready[0]['text_clean'])
        self.assertIn('Xin gửi tài liệu', ready[0]['text_clean'])

    def test_unverified_old_source_and_modified_text_fail(self):
        for row in (raw('new', post='1'), raw('new', post='3'), {**raw('new'), 'text': 'edited'}):
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.select([row], [decision(row)])

    def test_changed_context_evidence_and_duplicate_decisions_fail(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            directory = Path(temp)
            context = directory/'context.json'
            evidence = {key: raw('new')[key] for key in ('post_id', 'source_url')}
            context.write_text(json.dumps(evidence), encoding='utf-8')
            item = decision(raw('new')) | {'context_file': str(context), 'context_sha256': sha256(context.read_bytes())}
            path = directory/'labels.jsonl'; path.write_text(json.dumps(item)+'\n', encoding='utf-8')
            self.assertEqual(len(read_decisions([path])), 1)
            context.write_text('{"changed":true}', encoding='utf-8')
            with self.assertRaises(ValueError): read_decisions([path])
            item.pop('context_file'); item.pop('context_sha256')
            path.write_text((json.dumps(item)+'\n')*2, encoding='utf-8')
            with self.assertRaises(ValueError): read_decisions([path])

    def test_context_for_another_post_fails_even_with_valid_hash(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            directory = Path(temp)
            context = directory/'context.json'
            path = directory/'labels.jsonl'
            expected = {key: raw('new')[key] for key in ('post_id', 'source_url')}
            for evidence in ({**expected, 'post_id': '2k5ptit:3'},
                             {**expected, 'source_url': raw('new', post='3')['source_url']},
                             {}, []):
                with self.subTest(evidence=evidence):
                    context.write_text(json.dumps(evidence), encoding='utf-8')
                    item = decision(raw('new')) | {
                        'context_file': str(context), 'context_sha256': sha256(context.read_bytes())}
                    path.write_text(json.dumps(item)+'\n', encoding='utf-8')
                    with self.assertRaisesRegex(ValueError, 'Context source differs'):
                        read_decisions([path])


class ExpansionGateTests(unittest.TestCase):
    def test_audit_of_small_batch_cannot_create_a_training_set_and_frozen_drift_fails(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            directory = Path(temp)
            metadata = {'splits': {}}
            for number, name in enumerate(('train', 'val', 'test'), 10):
                row = {**OLD[0], 'id': name, 'post_id': f'old:{number}', 'text': f'old {name}',
                       'text_clean': f'old {name}', 'text_tokenized': f'old_{name}'}
                path = directory/f'{name}.csv'
                with path.open('w', newline='', encoding='utf-8-sig') as stream:
                    writer = csv.DictWriter(stream, fieldnames=TRAIN_FIELDS); writer.writeheader(); writer.writerow(row)
                metadata['splits'][name] = {'path': str(path), 'sha256': sha256(path.read_bytes()), 'samples': 1}
            manifest = directory/'manifest.json'; manifest.write_text(json.dumps(metadata), encoding='utf-8')
            row = raw('new message')
            source = directory/'sources.csv'
            source.write_text('post_id,url,observed_comments,count_evidence\n2,https://www.facebook.com/groups/2k5ptit/posts/2/,15,15 comments\n', encoding='utf-8')
            groups = directory/'groups.txt'; groups.write_text('https://www.facebook.com/groups/2k5ptit/\n', encoding='utf-8')
            collected = directory/'raw.csv'
            with collected.open('w', newline='', encoding='utf-8-sig') as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS); writer.writeheader(); writer.writerow(row)
            labels = directory/'decisions.jsonl'; labels.write_text(json.dumps(decision(row))+'\n', encoding='utf-8')
            output = directory/'expanded'
            args = ['--raw', str(collected), '--decisions', str(labels), '--sources', str(source),
                    '--groups', str(groups), '--frozen-manifest', str(manifest), '--output-dir', str(output),
                    '--report-dir', str(directory/'audit')]
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main(args + ['--audit-only']), 0)
                self.assertFalse(output.exists())
                self.assertEqual(main(args), 1)
                self.assertFalse(output.exists())
            (directory/'val.csv').write_text('modified', encoding='utf-8')
            with self.assertRaises(ValueError): load_frozen(manifest)


if __name__ == '__main__':
    unittest.main()
