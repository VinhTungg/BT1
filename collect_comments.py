"""Guided Facebook comment collection. Login and review remain interactive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

BASE = Path(__file__).resolve().parent
FIELDS = ["id", "post_id", "text", "label", "source_url", "collected_at", "extraction_method"]
DEFAULT_ARTICLE = '[role="article"]'
DEFAULT_TEXT = 'div[dir="auto"], span[dir="auto"]'
MORE = re.compile(
    r"^(?:xem (?:(?:thêm|các)\s+)?(?:\d+[.,]?\d*\s+)?bình luận(?: trước)?|"
    r"xem (?:(?:thêm|tất cả)\s+)?(?:\d+[.,]?\d*\s*)?(?:câu trả lời|phản hồi)|"
    r"xem thêm|view (?:\d+\s+)?(?:more|previous) comments|"
    r"view (?:\d+[.,]?\d*\s*)?(?:more )?repl(?:y|ies)|see more)"
    r"(?:\s.*|\s*\(.*\))?$",
    re.IGNORECASE,
)
COMMENT_FILTER = re.compile(r"^(?:Phù hợp nhất|Mới nhất|Most relevant|Newest)$", re.I)
ALL_COMMENTS = re.compile(r"^(?:Tất cả bình luận|All comments)(?:\s|$)", re.I)


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split())


def canonical_url(value: str) -> str:
    """Accept Facebook post permalinks, drop tracking and comment-specific queries."""
    url = urlsplit(value.strip())
    host = (url.hostname or "").lower()
    if (url.scheme != "https" or url.username or url.password or url.port or
            not (host == "facebook.com" or host.endswith(".facebook.com"))):
        raise ValueError("Cần link HTTPS trực tiếp trên facebook.com.")
    path = url.path.rstrip("/") + "/"
    if re.fullmatch(r"/groups/[^/]+/(?:posts|permalink)/[^/]+/", path):
        query = ""
    elif re.fullmatch(r"/share/p/[^/]+/", path):
        query = ""
    elif url.path in ("/permalink.php", "/story.php"):
        params = parse_qs(url.query)
        if not params.get("story_fbid") or not params.get("id"):
            raise ValueError("Link thiếu story_fbid hoặc id.")
        path = url.path
        query = urlencode({key: params[key][0] for key in ("story_fbid", "id")})
    else:
        raise ValueError("Cần link một bài đăng, ví dụ /groups/GROUP/posts/POST/; không phải link nhóm.")
    return urlunsplit(("https", "www.facebook.com", path, query, ""))


def post_key(url: str) -> str:
    match = re.search(r"/groups/([^/]+)/(?:posts|permalink)/([^/]+)", url)
    if match:
        return f"{match[1]}:{match[2]}"
    params = parse_qs(urlsplit(url).query)
    if params.get("story_fbid"):
        return f"{params['id'][0]}:{params['story_fbid'][0]}"
    return "url:" + hashlib.sha256(url.encode()).hexdigest()[:20]


def read_urls(path: Path) -> list[str]:
    urls = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            url = canonical_url(line)
        except ValueError as exc:
            raise ValueError(f"{path.name}, dòng {number}: {exc}") from exc
        if url not in urls:
            urls.append(url)
    if not urls:
        raise ValueError(f"Chưa có link bài đăng trong {path}.")
    return urls


class Dataset:
    """Append only; retain labels and deduplicate exact normalized text globally."""

    def __init__(self, path: Path, *, exclude_paths: list[Path] | None = None,
                 target_comments: int | None = None):
        self.path = path
        self.seen: set[str] = set()
        self.excluded: set[str] = set()
        self.target_comments = target_comments
        if target_comments is not None and target_comments < 1:
            raise ValueError("Mục tiêu số bình luận phải lớn hơn 0.")
        for source in exclude_paths or []:
            if source.resolve() == path.resolve():
                raise ValueError("Không dùng cùng file làm --output và --exclude-data.")
            with source.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames or "text" not in reader.fieldnames:
                    raise ValueError(f"File loại trùng thiếu cột text: {source}")
                for row in reader:
                    if row.get("text") is None or None in row:
                        raise ValueError(f"File loại trùng có dòng lỗi: {source}")
                    self.excluded.add(normalize_text(row["text"]).casefold())
        if path.exists() and path.stat().st_size:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != FIELDS:
                    raise ValueError(f"CSV {path} không đúng cấu trúc. Hãy chọn --output khác.")
                for row in reader:
                    if None in row or any(row.get(field) is None for field in FIELDS):
                        raise ValueError("CSV có dòng thiếu/thừa cột; cần kiểm tra trước khi ghi tiếp.")
                    self.seen.add(normalize_text(row["text"]))

    @property
    def target_reached(self) -> bool:
        return self.target_comments is not None and len(self.seen) >= self.target_comments

    def pending(self, candidates: list[dict]) -> list[dict]:
        seen = set(self.seen)
        result = []
        for candidate in candidates:
            text = normalize_text(candidate["text"])
            if text and text not in seen and text.casefold() not in self.excluded:
                result.append({**candidate, "text": text})
                seen.add(text)
        if self.target_comments is not None:
            result = result[:max(0, self.target_comments - len(self.seen))]
        return result

    def save(self, candidates: list[dict], source_url: str) -> int:
        fresh = self.pending(candidates)
        if not fresh:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        # Existing file already has its BOM. utf-8 avoids adding one mid-file.
        encoding = "utf-8-sig" if write_header else "utf-8"
        with self.path.open("a", encoding=encoding, newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            if write_header:
                writer.writeheader()
            for candidate in fresh:
                text = candidate["text"]
                writer.writerow({
                    "id": hashlib.sha256(text.encode()).hexdigest()[:24],
                    "post_id": post_key(source_url),
                    "text": text,
                    "label": "",
                    "source_url": source_url,
                    "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "extraction_method": candidate["extraction_method"],
                })
                handle.flush()
                self.seen.add(text)
        return len(fresh)


def check_access(page) -> None:
    path = urlsplit(page.url).path.lower()
    if any(part in path for part in ("/login", "/checkpoint", "/two_step_verification", "/recover",
                                     "/auth_platform", "/confirmemail")):
        raise ValueError("Facebook đang yêu cầu đăng nhập/xác minh. Tự xử lý trên trình duyệt trước.")
    login = page.locator('input[name="email"], input[name="pass"]')
    if any(login.nth(index).is_visible() for index in range(login.count())):
        raise ValueError("Bạn chưa đăng nhập hoặc phiên đăng nhập đã hết hạn.")


def browser_profile(browser: str, override: Path | None = None) -> Path:
    if override is not None:
        return override.resolve()
    # Keep the existing Chromium profile; use a separate profile for each channel.
    name = ".browser-profile" if browser == "chromium" else f".browser-profile-{browser}"
    return BASE / name


def wait_for_login(page) -> bool:
    """Wait for the human to finish login; never navigate away from a challenge."""
    print("Chương trình đang chờ bạn đăng nhập/xác minh trên trình duyệt.")
    print("Nếu trang xác minh trắng: thử tải lại một lần và hoàn tất kiểm tra khi nó hiện ra.")
    print("Chỉ tiếp tục khi bạn đã thấy Bảng tin hoặc bài đăng Facebook.")
    while True:
        command = input("Đã vào Facebook thì nhấn Enter; q để dừng và giữ profile: ").strip().lower()
        if command == "q":
            return False
        # Process pending browser events after the blocking terminal prompt.
        page.wait_for_timeout(250)
        try:
            check_access(page)
            host = (urlsplit(page.url).hostname or "").lower()
            if not (host == "facebook.com" or host.endswith(".facebook.com")):
                raise ValueError("Trang hiện tại chưa phải Facebook. Hãy mở Facebook và đăng nhập trước.")
        except ValueError as exc:
            # Never echo authentication URLs, cookies or verification tokens.
            print(f"Chưa thể tiếp tục: {exc}")
            print("Giữ cửa sổ này để tự hoàn tất xác minh; chương trình chưa mở bài đăng.")
            continue
        return True


def select_scope(page, args):
    if args.scope_selector:
        scopes = page.locator(args.scope_selector)
        visible = [scopes.nth(i) for i in range(scopes.count()) if scopes.nth(i).is_visible()]
        if len(visible) != 1:
            raise ValueError("--scope-selector phải khớp đúng một vùng hiển thị của bài đăng.")
        return visible[0]
    dialogs = page.get_by_role("dialog")
    for i in reversed(range(dialogs.count())):
        dialog = dialogs.nth(i)
        if dialog.is_visible():
            # A post dialog can exist before any comment has finished loading.
            post_heading = dialog.get_by_role("heading", name=re.compile(r"^(?:Bài viết của|Post by|.+['’]s [Pp]ost)"))
            if dialog.locator(args.article_selector).count() or post_heading.count():
                return dialog
    raise ValueError("Chưa tìm thấy hộp thoại bài đăng. Nếu bài nằm trên trang riêng, kiểm tra rồi đặt --scope-selector.")


def wait_for_post_scope(page, args):
    deadline = time.monotonic() + 15
    while True:
        check_access(page)
        try:
            return select_scope(page, args)
        except ValueError:
            if time.monotonic() >= deadline:
                raise
            page.wait_for_timeout(args.wait_ms)


def extract(page, args, script: str) -> list[dict]:
    check_access(page)
    return select_scope(page, args).evaluate(script, {
        "articleSelector": args.article_selector,
        "textSelector": args.text_selector,
        "customSelectors": args.article_selector != DEFAULT_ARTICLE or args.text_selector != DEFAULT_TEXT,
    })


def expand_once(page, args) -> bool:
    check_access(page)
    scope = select_scope(page, args)
    # Match only ordinary expand controls. Never click Like/Reply/Share/Send.
    for role in ("button",):
        controls = scope.get_by_role(role, name=MORE)
        for index in range(controls.count()):
            control = controls.nth(index)
            if control.is_visible() and control.is_enabled():
                control.click(timeout=5000)
                # A UI rendering wait, not an anti-detection delay.
                page.wait_for_timeout(getattr(args, "wait_ms", 1200))
                return True
    return False


def ensure_same_post(expected: str, current: str) -> str:
    """Accept numeric/name group aliases for the same post; reject another post."""
    expected = canonical_url(expected)
    current = canonical_url(current)
    if "/share/p/" in current:
        raise ValueError("Link chia sẻ chưa chuyển đến bài thật. Hãy dùng link trực tiếp của bài đăng.")
    if "/share/p/" not in expected and post_key(expected).rsplit(":", 1)[-1] != post_key(current).rsplit(":", 1)[-1]:
        raise ValueError("Trang đã chuyển sang bài khác với link bạn chọn. Dừng lấy để tránh lẫn nguồn.")
    return current


def select_all_comments(page, args) -> bool:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

    scope = select_scope(page, args)
    if scope.get_by_role("button", name=ALL_COMMENTS).count():
        return True
    filters = scope.get_by_role("button", name=COMMENT_FILTER)
    if not filters.count():
        try:
            filters.first.wait_for(state="visible", timeout=5000)
        except PlaywrightTimeoutError:
            check_access(page)
            if scope.get_by_role("button", name=ALL_COMMENTS).count():
                return True
    for index in range(filters.count()):
        control = filters.nth(index)
        if not control.is_visible():
            continue
        control.click(timeout=5000)
        choice = page.get_by_role("menuitem", name=ALL_COMMENTS)
        try:
            choice.first.wait_for(state="visible", timeout=5000)
        except PlaywrightTimeoutError:
            page.keyboard.press("Escape")
            check_access(page)
            raise ValueError("Không tìm thấy Tất cả bình luận. Hãy chọn bộ lọc bằng tay rồi thử lại.")
        choice.first.click(timeout=5000)
        page.wait_for_timeout(args.wait_ms)
        print("  Bộ lọc: Tất cả bình luận.", flush=True)
        return True
    print("  Không thấy bộ lọc bình luận; dùng phần bình luận hiện có trên bài.", flush=True)
    return False


def scroll_comments(page, args) -> None:
    scope = select_scope(page, args)
    articles = scope.locator(args.article_selector)
    if articles.count():
        articles.last.scroll_into_view_if_needed(timeout=5000)
    box = scope.bounding_box()
    if box:
        page.mouse.move(box["x"] + box["width"] * 0.65, box["y"] + box["height"] * 0.55)
        page.mouse.wheel(0, 900)
    page.wait_for_timeout(args.wait_ms)


def save_post_context(page, args, source: str) -> None:
    """Keep optional, bounded UI evidence for later context-dependent labels."""
    directory = getattr(args, 'context_dir', None)
    if directory is None:
        return
    ensure_same_post(source, page.url)
    text = select_scope(page, args).inner_text()
    ensure_same_post(source, page.url)
    directory.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(post_key(source).rsplit(':', 1)[-1].encode()).hexdigest()[:24]
    snapshot = {
        'source_url': source, 'post_id': post_key(source),
        'captured_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'context_kind': 'visible_post_scope_with_comments_and_ui',
        'max_chars': 24000, 'truncated': len(text) > 24000,
        'text': text[:24000],
    }
    serialized = json.dumps(snapshot, ensure_ascii=False, indent=2)
    first = directory / f'{key}_first.json'
    if not first.exists():
        temporary = first.with_suffix('.tmp')
        temporary.write_text(serialized, encoding='utf-8')
        temporary.replace(first)
    latest = directory / f'{key}_latest.json'
    temporary = latest.with_suffix('.tmp')
    temporary.write_text(serialized, encoding='utf-8')
    temporary.replace(latest)


def collect_post(page, args, script: str, dataset: Dataset, source: str, *, autosave: bool) -> dict:
    """Only collect the current user-selected post; no discovery or navigation."""
    candidates = {}
    saved = 0

    def capture():
        nonlocal saved
        check_access(page)
        ensure_same_post(source, page.url)
        records = extract(page, args, script)
        ensure_same_post(source, page.url)
        for record in records:
            text = normalize_text(record["text"])
            if text:
                candidates[text] = {**record, "text": text}
        if autosave:
            saved += dataset.save(list(candidates.values()), source)
        save_post_context(page, args, source)

    check_access(page)
    ensure_same_post(source, page.url)
    wait_for_post_scope(page, args)
    select_all_comments(page, args)
    page.wait_for_timeout(args.wait_ms)
    capture()
    if autosave and dataset.target_reached:
        print(f"  Đã đạt mục tiêu {dataset.target_comments} bình luận trong file đầu ra.", flush=True)
        return {"records": list(candidates.values()), "saved": saved, "rounds": 0, "stop_reason": "target_reached"}
    stable = 0
    stop_reason = "max_rounds"
    rounds = 0
    for rounds in range(1, args.max_rounds + 1):
        before = (len(candidates), select_scope(page, args).locator(args.article_selector).count())
        expanded = expand_once(page, args)
        if expanded:
            capture()
            if autosave and dataset.target_reached:
                stop_reason = "target_reached"
                break
        # Scrolling also loads top-level comments that have no "View more" button.
        scroll_comments(page, args)
        capture()
        if autosave and dataset.target_reached:
            stop_reason = "target_reached"
            break
        after = (len(candidates), select_scope(page, args).locator(args.article_selector).count())
        stable = stable + 1 if before == after else 0
        print(f"  Lượt {rounds}/{args.max_rounds}: nhận diện {len(candidates)} văn bản; "
              f"đã lưu mới {saved}; {'mở phản hồi + cuộn' if expanded else 'cuộn tải thêm'}.", flush=True)
        if stable >= args.idle_rounds:
            stop_reason = "no_more_loaded"
            break
    # One last rendering wait and capture before leaving the post.
    if stop_reason != "target_reached":
        page.wait_for_timeout(args.wait_ms)
        capture()
        if autosave and dataset.target_reached:
            stop_reason = "target_reached"
    message = {"no_more_loaded": "không tải thêm dữ liệu sau nhiều lượt",
               "target_reached": "đã đạt mục tiêu số bình luận",
               "max_rounds": "đã chạm giới hạn lượt; có thể còn bình luận chưa tải"}[stop_reason]
    print(f"  Dừng bài: {message}. Có {len(candidates)} văn bản khác nhau.", flush=True)
    return {"records": list(candidates.values()), "saved": saved, "rounds": rounds, "stop_reason": stop_reason}


def collect_and_review(page, args, script: str, dataset: Dataset, source: str) -> None:
    ensure_same_post(source, page.url)
    review_candidates(extract(page, args, script), dataset, source)


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Cần số nguyên lớn hơn 0.")
    return number


def main() -> int:
    # Windows terminals may otherwise use a code page that cannot print Vietnamese.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urls", type=Path, default=BASE / "post_urls.txt")
    parser.add_argument("--output", type=Path, default=BASE / "data/comments.csv")
    parser.add_argument("--browser", choices=("chromium", "chrome", "msedge"), default="chromium",
                        help="Dùng Chromium hoặc Chrome/Edge đã cài trên máy.")
    parser.add_argument("--profile", type=Path, help="Thư mục phiên riêng, mặc định tách theo trình duyệt.")
    parser.add_argument("--login-only", action="store_true",
                        help="Chỉ đăng nhập và lưu phiên, chưa mở bài/đọc dữ liệu.")
    parser.add_argument("--auto", action="store_true",
                        help="Tự mở thêm, cuộn và lưu từng đợt; chỉ dùng các link trong --urls.")
    parser.add_argument("--max-rounds", "--expand-clicks", dest="max_rounds", type=positive, default=120,
                        help="Giới hạn lượt mở thêm/cuộn cho mỗi bài (mặc định 120).")
    parser.add_argument("--idle-rounds", type=positive, default=3,
                        help="Dừng sau số lượt không có thêm dữ liệu (mặc định 3).")
    parser.add_argument("--wait-ms", type=positive, default=1200,
                        help="Thời gian chờ giao diện tải sau thao tác, mili giây (mặc định 1200).")
    parser.add_argument("--scope-selector", help="CSS selector vùng bài đăng nếu tự nhận diện sai.")
    parser.add_argument("--article-selector", default=DEFAULT_ARTICLE)
    parser.add_argument("--text-selector", default=DEFAULT_TEXT)
    parser.add_argument("--discover-groups", type=Path,
                        help="Chỉ tìm link bài từ file nhóm đã được cho phép; --urls là file đầu ra mới.")
    parser.add_argument("--discovery-report", type=Path, default=BASE / "reports/discovery/post_candidates.csv")
    parser.add_argument("--discovery-rounds", type=positive, default=40)
    parser.add_argument("--min-comments", type=positive, default=10)
    parser.add_argument("--max-posts", type=positive, default=150)
    parser.add_argument("--discovery-keyword", action="append", default=[],
                        help="Tìm thêm bài theo từ khóa trong từng nhóm đã cho phép; có thể lặp.")
    parser.add_argument("--discovery-search-only", action="store_true",
                        help="Chỉ duyệt kết quả tìm kiếm trong nhóm, không quét lại bảng tin.")
    parser.add_argument("--exclude-annotations", type=Path, default=BASE / "annotations")
    parser.add_argument("--exclude-data", type=Path, action="append", default=[],
                        help="CSV dữ liệu cũ có cột text; có thể lặp để không lưu lại nội dung cũ.")
    parser.add_argument("--target-comments", type=positive,
                        help="Dừng khi CSV đầu ra có đủ số bình luận duy nhất; không tính file --exclude-data.")
    parser.add_argument("--context-dir", type=Path,
                        help="Lưu riêng bản chữ của vùng bài đang mở để rà nhãn cần ngữ cảnh; có thể gồm bình luận và UI.")
    args = parser.parse_args()
    profile = browser_profile(args.browser, args.profile)
    urls = []
    dataset = None
    script = ""
    try:
        if args.discover_groups and not args.login_only:
            from src.post_discovery import read_groups, known_post_ids, discovery_sources
            urls = read_groups(args.discover_groups)
            known_post_ids(args.exclude_annotations)
            discovery_sources(urls, args.discovery_keyword, args.discovery_search_only)
            if args.urls.exists():
                raise ValueError("Chọn --urls là file đầu ra mới khi tìm link, tránh ghi đè danh sách đang có.")
        elif not args.login_only:
            urls = read_urls(args.urls)
            dataset = Dataset(args.output, exclude_paths=args.exclude_data, target_comments=args.target_comments)
            script = (BASE / "extract_comments.js").read_text(encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 1
    try:
        from playwright.sync_api import Error as PlaywrightError, TimeoutError as PlaywrightTimeoutError, sync_playwright
    except ImportError:
        print("Thiếu Playwright. Chạy: python -m pip install -r requirements.txt\n"
              "Sau đó: python -m playwright install chromium", file=sys.stderr)
        return 1
    if dataset is not None:
        print(f"Có {len(urls)} bài; CSV hiện có {len(dataset.seen)} nội dung khác nhau.")
    print(f"Trình duyệt: {args.browser}; profile: {profile}")
    print("Dùng một profile riêng; bạn tự đăng nhập trong cửa sổ trình duyệt.")
    print("Chỉ thu thập nguồn bạn có quyền sử dụng. Nhãn luôn để trống để bạn gán sau.")
    try:
        print("[1/3] Khởi tạo Playwright...", flush=True)
        with sync_playwright() as playwright:
            print("[2/3] Mở trình duyệt (tối đa 45 giây)...", flush=True)
            context = playwright.chromium.launch_persistent_context(
                str(profile), channel=None if args.browser == "chromium" else args.browser,
                headless=False, viewport={"width": 1360, "height": 900}, timeout=45000,
            )
            try:
                page = context.pages[0] if context.pages else context.new_page()
                page.set_default_timeout(10000)
                print("[3/3] Mở Facebook (tối đa 60 giây)...", flush=True)
                try:
                    page.goto("https://www.facebook.com/", wait_until="domcontentloaded", timeout=60000)
                except PlaywrightTimeoutError:
                    print("Trang tải chậm. Giữ trình duyệt mở để bạn kiểm tra và tải lại thủ công.")
                if not wait_for_login(page):
                    return 0
                if args.login_only:
                    print(f"Đã kết thúc bước đăng nhập; phiên được giữ trong {profile}")
                    print(f"Lần sau chạy lại với --browser {args.browser} để thu thập.")
                    return 0
                if args.discover_groups:
                    from src.post_discovery import run_discovery
                    try:
                        result = run_discovery(
                            page, groups=urls, annotation_dir=args.exclude_annotations,
                            output_urls=args.urls, report_path=args.discovery_report,
                            rounds=args.discovery_rounds, minimum=args.min_comments,
                            max_posts=args.max_posts, wait_ms=args.wait_ms, check_access=check_access,
                            keywords=args.discovery_keyword, search_only=args.discovery_search_only,
                        )
                    except (PlaywrightError, ValueError, OSError) as exc:
                        print(f"Dừng tìm link: {exc}; các link đã tìm được vẫn được giữ.", file=sys.stderr)
                        return 1
                    print(f"Đã lưu {result['accepted_posts']} link bài đủ điều kiện vào {args.urls}")
                    return 0
                for index, source in enumerate(urls, 1):
                    if dataset.target_reached:
                        print(f"Đã đủ {len(dataset.seen)} bình luận trong {args.output}; dừng thu thập.", flush=True)
                        break
                    print(f"\nBài {index}/{len(urls)}: {source}")
                    try:
                        page.goto(source, wait_until="domcontentloaded", timeout=60000)
                        page.wait_for_timeout(args.wait_ms)
                        try:
                            check_access(page)
                        except ValueError:
                            print("Facebook yêu cầu đăng nhập/xác minh lại trước khi mở bài.")
                            if not wait_for_login(page):
                                return 0
                            page.goto(source, wait_until="domcontentloaded", timeout=60000)
                            check_access(page)
                        resolved = ensure_same_post(source, page.url)
                    except (PlaywrightError, ValueError) as exc:
                        print(f"Không mở được bài: {exc}\nBỏ qua bài này; CSV trước đó vẫn được giữ.")
                        continue
                    if args.auto:
                        try:
                            collect_post(page, args, script, dataset, resolved, autosave=True)
                        except (PlaywrightError, ValueError, OSError) as exc:
                            print(f"Dừng bài hiện tại: {exc}\nCác đợt đã lưu vẫn nằm trong CSV.", flush=True)
                        continue
                    print("Trên trình duyệt: chọn Tất cả bình luận (nếu có), mở phản hồi, cuộn để tải.")
                    print("Enter: lấy + xem trước | a: tự mở thêm/cuộn rồi xem trước | n: bài tiếp | q: thoát")
                    while True:
                        command = input("> ").strip().lower()
                        if command == "q":
                            return 0
                        if command == "n":
                            break
                        try:
                            ensure_same_post(resolved, page.url)
                            if command == "a":
                                result = collect_post(page, args, script, dataset, resolved, autosave=False)
                                review_candidates(result["records"], dataset, resolved)
                            elif command == "":
                                collect_and_review(page, args, script, dataset, resolved)
                            else:
                                print("Chọn Enter, a, n hoặc q.")
                        except (PlaywrightError, ValueError, OSError) as exc:
                            print(f"Chưa lấy/lưu được: {exc}\nKiểm tra trang và file CSV; thử lại hoặc chọn n/q.")
            finally:
                context.close()
    except (KeyboardInterrupt, EOFError):
        print("\nĐã dừng. Những dòng đã lưu vẫn nằm trong CSV.")
    except PlaywrightError as exc:
        print(f"Lỗi trình duyệt: {exc}", file=sys.stderr)
        if args.browser == "chromium":
            print("Nếu chưa cài Chromium: python -m playwright install chromium", file=sys.stderr)
        else:
            print("Kiểm tra Chrome/Edge đã cài và không có chương trình khác dùng cùng --profile.", file=sys.stderr)
        return 1
    return 0


def review_candidates(candidates: list[dict], dataset: Dataset, source: str) -> None:
    pending = dataset.pending(candidates)
    print(f"Nhận diện {len(candidates)} ứng viên; {len(pending)} nội dung mới sau loại trùng.")
    if not pending:
        print("Nếu trang có bình luận, hãy mở thêm hoặc điều chỉnh selector. Không có dữ liệu được tạo giả.")
        return
    print("XEM TRƯỚC — đối chiếu với bài đăng để loại trường hợp nhận nhầm:")
    for index, item in enumerate(pending[:10], 1):
        print(f"  {index}. [{item['extraction_method']}] {item['text']}")
    if pending and input(f"Gõ s để lưu {len(pending)} dòng; Enter để bỏ: ").strip().lower() == "s":
        print(f"Đã lưu {dataset.save(pending, source)} dòng vào {dataset.path}")


if __name__ == "__main__":
    raise SystemExit(main())
