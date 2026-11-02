"""Source discovery tests: never collect posts without visible count evidence."""
import csv
import tempfile
import unittest
from pathlib import Path

from src.post_discovery import accept_candidates, canonical_post, comment_count, group_url, known_post_ids, discovery_sources, observe_search_post, EXTRACT


class PostDiscoveryTests(unittest.TestCase):
    def test_comment_counts_and_ambiguous_numbers(self):
        cases = {
            "10 bình luận": 10, "9 comments": 9, "1,2K bình luận": 1200,
            "1.234 bình luận": 1234, "2 nghìn bình luận": 2000,
            "10+ bình luận": 10, "View all 52 comments": 52,
            "Comment on Người tham gia ẩn danh's post, 51 comments": 51,
            "Comment on Example's post, 1 comment": 1,
            "1,2 bình luận": None, "100 lượt thích": None,
            "Bài này có 15 bình luận thật vui": None, "Xem 20 phản hồi": None,
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(comment_count(value), expected)

    def test_only_explicit_group_links_and_observed_post_ids(self):
        self.assertEqual(group_url("https://www.facebook.com/groups/2k5ptit"), "https://www.facebook.com/groups/2k5ptit/")
        for value in ("https://facebook.com.evil.test/groups/2k5ptit", "http://facebook.com/groups/2k5ptit", "https://facebook.com/groups/2k5ptit/posts/123"):
            with self.assertRaises(ValueError): group_url(value)
        self.assertEqual(canonical_post("https://www.facebook.com/groups/2k5ptit/permalink/123/?comment_id=9", {"2k5ptit"}), ("123", "https://www.facebook.com/groups/2k5ptit/posts/123/"))
        self.assertEqual(canonical_post("https://www.facebook.com/groups/2k5ptit/?multi_permalinks=456", {"2k5ptit"})[0], "456")
        self.assertEqual(canonical_post("https://www.facebook.com/groups/2k5ptit/search/?q=x&multi_permalinks=456", {"2k5ptit"})[0], "456")
        self.assertIsNone(canonical_post("https://www.facebook.com/groups/other/posts/123/", {"2k5ptit"}))
        self.assertIsNone(canonical_post("https://www.facebook.com/share/p/unknown/", {"2k5ptit"}))

    def test_reject_old_posts_small_posts_and_unverified_counts(self):
        records = [
            {"url": "https://www.facebook.com/groups/2k5ptit/posts/123/", "labels": ["500 bình luận"]},
            {"url": "https://www.facebook.com/groups/2k5ptit/posts/456/", "labels": ["9 bình luận"]},
            {"url": "https://www.facebook.com/groups/2k5ptit/posts/789/", "labels": ["12 bình luận", "25 lượt thích"]},
            {"url": "https://www.facebook.com/groups/other/posts/999/", "labels": ["999 bình luận"]},
            {"url": "https://www.facebook.com/groups/2k5ptit/posts/888/", "labels": ["Bình luận"]},
        ]
        result = accept_candidates(records, {"2k5ptit"}, {"123"}, 10)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["post_id"], "789")
        self.assertEqual(result[0]["observed_comments"], 12)

    def test_exclusions_use_post_id_across_group_aliases(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "thanh_vien_1.csv"
            path.write_text("post_id\n2k5ptit:123\n98765:123\nconfessions.ptit:456\n", encoding="utf-8")
            self.assertEqual(known_post_ids(Path(temp)), {"123", "456"})

    def test_keyword_search_stays_inside_explicit_groups(self):
        groups = ["https://www.facebook.com/groups/2k5ptit/", "https://www.facebook.com/groups/123/"]
        tasks = discovery_sources(groups, ["phốt", "phốt", "học phí"], search_only=True)
        self.assertEqual(len(tasks), 4)
        self.assertTrue(all(url.startswith(source + "search/?q=") for source, url in tasks))
        self.assertEqual({source for source, _ in tasks}, set(groups))
        with self.assertRaises(ValueError):
            discovery_sources(groups, [], search_only=True)

    def test_extract_count_and_link_from_same_article_without_nested_leakage(self):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            self.skipTest("Playwright unavailable")
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(headless=True, channel="chrome")
            except Exception as exc:
                self.skipTest(f"Chrome unavailable: {exc}")
            try:
                page = browser.new_page()
                page.set_content('''
                <div role="article"><a href="https://www.facebook.com/groups/2k5ptit/posts/100/">time</a><span>15 bình luận</span>
                  <div role="article"><a href="https://www.facebook.com/groups/2k5ptit/posts/200/">nested</a><span>2 bình luận</span></div>
                </div>
                <div role="article"><a href="https://www.facebook.com/groups/2k5ptit/posts/300/">time</a><span style="display:none">999 bình luận</span><span>1 bình luận</span></div>
                <div class="post"><div role="article"><a href="https://www.facebook.com/groups/2k5ptit/posts/400/">header only</a></div><div><div>22 bình luận</div></div></div>
                ''')
                raw = page.evaluate(EXTRACT)
                candidates = accept_candidates(raw, {"2k5ptit"}, set(), 10)
                self.assertEqual({r["post_id"] for r in candidates}, {"100", "400"})
                self.assertEqual(next(r for r in candidates if r["post_id"] == "100")["observed_comments"], 15)
                self.assertEqual(next(r for r in candidates if r["post_id"] == "400")["observed_comments"], 22)
            finally:
                browser.close()

    def test_search_card_open_observes_matching_post_and_preserves_source(self):
        from playwright.sync_api import sync_playwright
        search = 'https://www.facebook.com/groups/2k5ptit/search/?q=test'
        html = '''<button aria-label="Comment on Small's post, 9 comments" onclick="location.href='/groups/2k5ptit/posts/9/'">9</button>
        <button aria-label="Comment on Larger's post, 133 comments" onclick="location.href='/groups/2k5ptit/posts/133/'">133</button>
        <button aria-label="Comment on Other's post, 20 comments" onclick="location.href='/groups/2k5ptit/posts/20/'">20</button>'''
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, channel='chrome')
            page = browser.new_page()
            page.route('**/*', lambda route: route.fulfill(content_type='text/html', body=html))
            try:
                page.goto(search)
                with tempfile.TemporaryDirectory() as temp:
                    tried = set()
                    raw = observe_search_post(page, {'2k5ptit'}, 10, tried, wait_ms=20,
                                              check_access=lambda page: None, report_path=Path(temp)/'report.csv')
                    rows = accept_candidates(raw, {'2k5ptit'}, set(), 10)
                    self.assertEqual([(r['post_id'], r['observed_comments']) for r in rows], [('133', 133)])
                    self.assertEqual(page.url, search)
                    self.assertEqual(tried, {"Comment on Larger's post, 133 comments"})
                    from collect_comments import check_access
                    page.set_content('''<button aria-label="Comment on Test's post, 100 comments" onclick="location.href='/checkpoint/'">100</button>''')
                    with self.assertRaises(ValueError):
                        observe_search_post(page, {'2k5ptit'}, 10, set(), wait_ms=20,
                                            check_access=check_access, report_path=Path(temp)/'report.csv')
                    self.assertEqual(page.url, 'https://www.facebook.com/checkpoint/')
            finally:
                browser.close()


if __name__ == "__main__":
    unittest.main()
