"""Local synthetic HTML only; all browser requests are intercepted."""

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from collect_comments import BASE, DEFAULT_ARTICLE, DEFAULT_TEXT, Dataset, collect_post, ensure_same_post, select_scope

POST = 'https://www.facebook.com/groups/test/posts/123/'
HTML = '''<!doctype html><html><head><meta charset="utf-8"></head><body>
<main><div role="article" aria-label="Comment by Background">
<div dir="auto">Không được lấy bình luận ngoài hộp thoại.</div><button>Reply</button></div></main>
<div role="dialog" aria-label="Bài viết của Người kiểm thử" style="height:400px;overflow:auto">
<h2>Bài viết của Người kiểm thử</h2>
<button id="filter" onclick="document.getElementById('menu').hidden=false">Phù hợp nhất</button>
<div id="menu" role="menu" hidden><button role="menuitem" onclick="document.getElementById('filter').textContent='Tất cả bình luận';document.getElementById('menu').hidden=true;window.allSelected=true">Tất cả bình luận Hiển thị tất cả bình luận.</button></div>
<div id="comments"><div role="article" aria-label="Bình luận dưới tên A vào 1 giờ trước">
<a><span dir="auto">Tác giả A</span></a><div dir="auto">Bình luận ban đầu.</div>
<button onclick="window.wrong=true">Thích</button><button onclick="window.wrong=true">Trả lời</button>
</div></div>
<button id="more" onclick="this.remove();setTimeout(()=>{document.getElementById('comments').insertAdjacentHTML('beforeend', '&lt;div role=&quot;article&quot; aria-label=&quot;A đáp lại phản hồi của B vào 1 giờ trước&quot;&gt;&lt;a&gt;&lt;span dir=&quot;auto&quot;&gt;Tác giả B&lt;/span&gt;&lt;/a&gt;&lt;div dir=&quot;auto&quot;&gt;&lt;a href=&quot;/groups/test/user/9/&quot;&gt;Tên được nhắc&lt;/a&gt; nội dung phản hồi.&lt;/div&gt;&lt;button&gt;Trả lời&lt;/button&gt;&lt;/div&gt;')},20)">Xem 1 phản hồi</button>
</div></body></html>'''


class ScopeTests(unittest.TestCase):
    def test_aliases_allowed_but_another_post_is_rejected(self):
        self.assertEqual(ensure_same_post(POST, POST.replace('/test/', '/999/')), POST.replace('/test/', '/999/'))
        with self.assertRaises(ValueError):
            ensure_same_post(POST, POST.replace('/123/', '/456/'))
        with self.assertRaises(ValueError):
            ensure_same_post(POST, 'https://www.facebook.com/groups/test/')


class AutoCollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise unittest.SkipTest('Playwright chưa được cài.')
        cls.playwright = sync_playwright().start()
        try:
            cls.browser = cls.playwright.chromium.launch(headless=True, channel='chromium')
        except Exception as exc:
            cls.playwright.stop()
            if "Executable doesn't exist" in str(exc):
                raise unittest.SkipTest('Chưa cài Chromium cho Playwright.')
            raise

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()

    def run_fixture(self, *, autosave=True, max_rounds=6, html=HTML, target_comments=None, save_context=False):
        page = self.browser.new_page()
        page.route('**/*', lambda route: route.fulfill(status=200, content_type='text/html; charset=utf-8', body=html))
        try:
            page.goto(POST)
            args = SimpleNamespace(scope_selector=None, article_selector=DEFAULT_ARTICLE,
                                   text_selector=DEFAULT_TEXT, wait_ms=80,
                                   max_rounds=max_rounds, idle_rounds=2)
            with tempfile.TemporaryDirectory(dir=BASE) as directory:
                output = Path(directory) / 'comments.csv'
                if save_context:
                    args.context_dir = Path(directory) / 'contexts'
                dataset = Dataset(output, target_comments=target_comments)
                with redirect_stdout(io.StringIO()):
                    result = collect_post(page, args, (BASE/'extract_comments.js').read_text(encoding='utf-8'),
                                          dataset, POST, autosave=autosave)
                rows = []
                if output.exists():
                    with output.open(encoding='utf-8-sig', newline='') as handle:
                        rows = list(csv.DictReader(handle))
                if save_context:
                    result['context_snapshots'] = {p.stem.rsplit('_', 1)[-1]: json.loads(p.read_text(encoding='utf-8'))
                                                   for p in args.context_dir.glob('*.json')}
                return result, rows, page.evaluate('Boolean(window.allSelected)'), page.evaluate('Boolean(window.wrong)')
        finally:
            page.close()

    def test_autosave_filter_delayed_reply_and_scope(self):
        result, rows, selected, wrong = self.run_fixture()
        self.assertTrue(selected)
        self.assertFalse(wrong)
        self.assertEqual({row['text'] for row in rows}, {'Bình luận ban đầu.', '[MENTION] nội dung phản hồi.'})
        self.assertTrue(all(row['label'] == '' and row['source_url'] == POST for row in rows))
        self.assertEqual(result['stop_reason'], 'no_more_loaded')

    def test_preview_does_not_write_until_user_saves(self):
        result, rows, _, _ = self.run_fixture(autosave=False)
        self.assertEqual(rows, [])
        self.assertEqual(len(result['records']), 2)

    def test_target_stops_before_expanding_more_comments(self):
        result, rows, _, wrong = self.run_fixture(target_comments=1)
        self.assertEqual(result['stop_reason'], 'target_reached')
        self.assertEqual(result['rounds'], 0)
        self.assertEqual(len(rows), 1)
        self.assertFalse(wrong)

    def test_context_evidence_keeps_first_and_latest_of_the_same_post_only(self):
        result, rows, _, _ = self.run_fixture(save_context=True)
        snapshots = result['context_snapshots']
        self.assertEqual(set(snapshots), {'first', 'latest'})
        for snapshot in snapshots.values():
            self.assertEqual(snapshot['source_url'], POST)
            self.assertEqual(snapshot['context_kind'], 'visible_post_scope_with_comments_and_ui')
            self.assertNotIn('Không được lấy bình luận ngoài hộp thoại.', snapshot['text'])
        self.assertNotIn('nội dung phản hồi.', snapshots['first']['text'])
        self.assertIn('nội dung phản hồi.', snapshots['latest']['text'])
        self.assertEqual(len(rows), 2)

    def test_max_rounds_is_enforced(self):
        result, _, _, _ = self.run_fixture(max_rounds=1)
        self.assertEqual(result['rounds'], 1)
        self.assertEqual(result['stop_reason'], 'max_rounds')

    def test_no_background_feed_fallback_when_post_dialog_missing(self):
        page = self.browser.new_page()
        try:
            page.set_content('<main role="main"><div role="article" aria-label="Comment by Other"><div dir="auto">Bài khác</div></div></main>')
            args = SimpleNamespace(scope_selector=None, article_selector=DEFAULT_ARTICLE)
            with self.assertRaises(ValueError):
                select_scope(page, args)
        finally:
            page.close()

    def test_another_post_is_rejected_before_any_save(self):
        page = self.browser.new_page()
        page.route('**/*', lambda route: route.fulfill(status=200, content_type='text/html; charset=utf-8', body=HTML))
        try:
            page.goto(POST.replace('/123/', '/456/'))
            args = SimpleNamespace(scope_selector=None, article_selector=DEFAULT_ARTICLE,
                                   text_selector=DEFAULT_TEXT, wait_ms=80, max_rounds=3, idle_rounds=2)
            with tempfile.TemporaryDirectory(dir=BASE) as directory:
                output = Path(directory) / 'comments.csv'
                with self.assertRaises(ValueError):
                    collect_post(page, args, (BASE/'extract_comments.js').read_text(encoding='utf-8'),
                                 Dataset(output), POST, autosave=True)
                self.assertFalse(output.exists())
        finally:
            page.close()


if __name__ == '__main__':
    unittest.main()
