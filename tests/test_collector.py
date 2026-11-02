"""Synthetic data only. No Facebook requests or login needed."""

import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from collect_comments import (BASE, DEFAULT_ARTICLE, DEFAULT_TEXT, FIELDS, MORE, Dataset,
                              browser_profile, canonical_url, expand_once, post_key, read_urls, wait_for_login)

URL = "https://www.facebook.com/groups/123/posts/456/"


class CollectorTests(unittest.TestCase):
    def test_verification_blocks_continuation_without_exposing_tokens(self):
        page = SimpleNamespace(
            url="https://www.facebook.com/two_step_verification/authentication/?encrypted_context=PRIVATE_TOKEN",
            wait_for_timeout=Mock(),
        )
        output = io.StringIO()
        with patch("builtins.input", side_effect=["", "q"]), redirect_stdout(output):
            self.assertFalse(wait_for_login(page))
        self.assertIn("Chưa thể tiếp tục", output.getvalue())
        self.assertNotIn("PRIVATE_TOKEN", output.getvalue())

    def test_verification_can_finish_manually_then_continue(self):
        page = SimpleNamespace(
            url="https://www.facebook.com/checkpoint/",
            wait_for_timeout=Mock(), locator=Mock(return_value=Mock(count=Mock(return_value=0))),
        )
        commands = iter(["", ""])

        def manual_input(_prompt):
            value = next(commands)
            if page.wait_for_timeout.call_count:
                page.url = "https://www.facebook.com/"
            return value

        with patch("builtins.input", side_effect=manual_input), redirect_stdout(io.StringIO()):
            self.assertTrue(wait_for_login(page))
        self.assertEqual(page.wait_for_timeout.call_count, 2)

    def test_browser_profiles_are_separate_and_keep_existing_chromium(self):
        self.assertEqual(browser_profile("chromium"), BASE / ".browser-profile")
        self.assertNotEqual(browser_profile("chrome"), browser_profile("msedge"))
        self.assertNotEqual(browser_profile("chrome"), browser_profile("chromium"))
        self.assertEqual(browser_profile("chrome", Path("custom-profile")), Path("custom-profile").resolve())

    def test_urls_drop_tracking_and_reject_other_hosts_or_group_home(self):
        self.assertEqual(canonical_url(URL + "?comment_id=9&__cft__=secret#anchor"), URL)
        self.assertEqual(canonical_url(URL.replace("www.", "m.")), URL)
        self.assertEqual(post_key(URL), "123:456")
        for url in ["https://facebook.com.evil.test/groups/1/posts/2/",
                    "https://www.facebook.com/groups/123/", "http://www.facebook.com/groups/1/posts/2/",
                    "https://user:pass@www.facebook.com/groups/1/posts/2/"]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                canonical_url(url)

    def test_input_and_global_duplicates_resume_preserve_labels(self):
        with tempfile.TemporaryDirectory(dir=BASE) as temp:
            root = Path(temp)
            urls_file = root / "urls.txt"
            urls_file.write_text(f"# ghi chú\n{URL}\n{URL}?comment_id=1\n", encoding="utf-8-sig")
            self.assertEqual(read_urls(urls_file), [URL])
            path = root / "comments.csv"
            data = Dataset(path)
            candidates = [{"text": 'Câu thử, có "dấu"\n tiếng Việt', "extraction_method": "fixture"}]
            self.assertEqual(data.save(candidates + candidates, URL), 1)
            with path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["label"] = "1"
            with path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            resumed = Dataset(path)
            self.assertEqual(resumed.save(candidates, URL.replace("456", "789")), 0)
            self.assertEqual(resumed.save([{"text": "Câu khác", "extraction_method": "fixture"}], URL), 1)
            with path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["label"], "1")
            self.assertEqual(rows[1]["label"], "")
            self.assertEqual(rows[0]["text"], 'Câu thử, có "dấu" tiếng Việt')

    def test_reject_incompatible_csv(self):
        with tempfile.TemporaryDirectory(dir=BASE) as temp:
            path = Path(temp) / "bad.csv"
            path.write_text("text,label\nhello,0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                Dataset(path)

    def test_new_batch_excludes_old_text_casefold_and_caps_only_new_rows(self):
        with tempfile.TemporaryDirectory(dir=BASE) as temp:
            root = Path(temp)
            old = root / "old.csv"
            old.write_text("text,label\nCÂU CŨ,0\n", encoding="utf-8-sig")
            output = root / "new.csv"
            dataset = Dataset(output, exclude_paths=[old], target_comments=2)
            candidates = [{"text": text, "extraction_method": "fixture"} for text in
                          ("câu cũ", "Câu mới một", "Câu mới hai", "Câu mới ba")]
            self.assertEqual(dataset.save(candidates, URL), 2)
            self.assertTrue(dataset.target_reached)
            with output.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([r["text"] for r in rows], ["Câu mới một", "Câu mới hai"])
            self.assertEqual(dataset.save(candidates, URL), 0)
            resumed = Dataset(output, exclude_paths=[old], target_comments=3)
            self.assertEqual(resumed.save(candidates, URL), 1)
            self.assertTrue(resumed.target_reached)

    def test_exclusion_file_errors_do_not_create_output(self):
        with tempfile.TemporaryDirectory(dir=BASE) as temp:
            root = Path(temp)
            old = root / "old.csv"
            old.write_text("label\n0\n", encoding="utf-8")
            output = root / "new.csv"
            with self.assertRaises(ValueError):
                Dataset(output, exclude_paths=[old])
            self.assertFalse(output.exists())
            with self.assertRaises(ValueError):
                Dataset(old, exclude_paths=[old])

    def test_expand_labels_only(self):
        for label in ["Xem thêm bình luận", "Xem thêm 10 bình luận", "Xem 3 phản hồi",
                      "Xem tất cả 3 phản hồi", "View more comments", "View 2 replies", "See more"]:
            with self.subTest(label=label):
                self.assertRegex(label, MORE)
        for label in ["Thích", "Trả lời", "Chia sẻ", "Send", "Reply", "Join group"]:
            self.assertNotRegex(label, MORE)


class ExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise unittest.SkipTest("Playwright chưa được cài.")
        cls.playwright = sync_playwright().start()
        try:
            cls.browser = cls.playwright.chromium.launch(headless=True, channel="chromium")
        except Exception as exc:
            cls.playwright.stop()
            if "Executable doesn't exist" in str(exc):
                raise unittest.SkipTest(f"Chưa có Chromium khả dụng: {exc}")
            raise

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()

    def test_nested_comments_no_author_post_or_action_text(self):
        page = self.browser.new_page()
        try:
            page.set_content('''<main role="main"><div role="article">
                <a><span dir="auto">Tác giả bài viết</span></a>
                <div dir="auto">Nội dung bài đăng, không phải bình luận.</div>
                <div role="article" aria-label="Bình luận của Người A">
                    <a><span dir="auto">Người A</span></a>
                    <div dir="auto"><span dir="auto">Môn này khó quá.</span></div>
                    <button>Thích</button><button>Trả lời</button>
                    <div role="article" aria-label="Reply by Person B">
                        <a><span dir="auto">Person B</span></a>
                        <div dir="auto">Mình cũng thấy vậy.</div><button>Reply</button>
                    </div>
                </div>
                <div role="article"><a><span dir="auto">Người C</span></a>
                    <div dir="auto">Lịch thi ở đâu nhỉ?</div><button>Trả lời</button>
                </div>
                <div role="article" aria-label="Comment by Hidden" style="display:none">
                    <div dir="auto">Ẩn, không lấy.</div>
                </div>
            </div></main>''')
            script = (BASE / "extract_comments.js").read_text(encoding="utf-8")
            result = page.get_by_role("main").evaluate(script, {
                "articleSelector": DEFAULT_ARTICLE, "textSelector": DEFAULT_TEXT,
                "customSelectors": False,
            })
            self.assertEqual([item["text"] for item in result],
                             ["Môn này khó quá.", "Mình cũng thấy vậy.", "Lịch thi ở đâu nhỉ?"])
        finally:
            page.close()

    def test_expansion_never_clicks_reply_or_like(self):
        from types import SimpleNamespace

        page = self.browser.new_page()
        try:
            page.set_content('''<main role="main">
                <button onclick="window.wrong = true">Thích</button>
                <button onclick="window.wrong = true">Trả lời</button>
                <button onclick="window.expanded = true; this.remove()">Xem thêm bình luận</button>
            </main>''')
            args = SimpleNamespace(scope_selector='main', article_selector=DEFAULT_ARTICLE)
            self.assertTrue(expand_once(page, args))
            self.assertTrue(page.evaluate("Boolean(window.expanded)"))
            self.assertFalse(page.evaluate("Boolean(window.wrong)"))
            self.assertFalse(expand_once(page, args))
        finally:
            page.close()

    def test_vietnamese_comment_label_and_inline_mention_preserve_sentence(self):
        page = self.browser.new_page()
        try:
            page.set_content('''<main role="main">
                <div role="article" aria-label="Bình luận dưới tên Người A vào 1 giờ trước">
                    <a><span dir="auto">Người A</span></a>
                    <span dir="auto"><div dir="auto"><a href="/groups/123/user/456/">Người B</a> câu thử có gắn thẻ <span role="link">Người ẩn danh</span></div></span>
                    <button>Trả lời</button>
                </div>
                <div role="article" aria-label="Bình luận dưới tên C vào 1 giờ trước">
                    <div dir="auto"><a href="/groups/123/user/789/">Chỉ gắn thẻ</a></div>
                </div>
            </main>''')
            result = page.get_by_role("main").evaluate(
                (BASE / "extract_comments.js").read_text(encoding="utf-8"),
                {"articleSelector": DEFAULT_ARTICLE, "textSelector": DEFAULT_TEXT, "customSelectors": False},
            )
            self.assertEqual(result, [{"text": "[MENTION] câu thử có gắn thẻ [MENTION]", "extraction_method": "comment_aria_label"}])
        finally:
            page.close()


if __name__ == "__main__":
    unittest.main()
