"""Discover post permalinks on explicitly authorized Facebook group feeds.

Only observed links with an observed comment-count label are accepted. No
requests to private APIs, no generated post IDs, and no comment extraction.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from urllib.parse import parse_qs, urlsplit, quote

COUNT = re.compile(
    r"^\s*(?:Xem tất cả\s+|View all\s+)?(?P<number>\d[\d.,]*)\s*"
    r"(?P<unit>k|m|nghìn|ngàn|triệu)?\+?\s*(?:lượt\s+)?(?:bình luận|comments?)\s*$", re.I,
)
FIELDS = ["post_id", "url", "group_url", "observed_comments", "count_evidence", "post_excerpt"]
EXTRACT = r"""() => {
  const result = [];
  const counterCache = new Map();
  function labelsIn(root, ownArticleOnly = false) {
    const key = ownArticleOnly ? null : root;
    if (key && counterCache.has(key)) return counterCache.get(key);
    const labels = new Set();
    for (const node of [root, ...root.querySelectorAll('div,a,button,[role="button"],span')]) {
      if (!node.getClientRects().length) continue;
      if (ownArticleOnly && node.closest('[role="article"]') !== root) continue;
      for (const value of [node.getAttribute('aria-label'), node.innerText]) {
        if (value && value.length < 100 && /\d.*(?:bình luận|comments?)/i.test(value)) labels.add(value.trim());
      }
    }
    const values = [...labels];
    if (key) counterCache.set(key, values);
    return values;
  }
  function postId(href) {
    const match = href.match(/\/groups\/[^/]+\/(?:posts|permalink)\/(\d+)/) || href.match(/[?&]multi_permalinks=(\d+)/);
    return match ? match[1] : null;
  }
  for (const article of document.querySelectorAll('[role="article"]')) {
    const labels = labelsIn(article, true);
    if (!labels.length) continue;
    const links = Array.from(article.querySelectorAll('a[href]'))
      .filter(a => a.closest('[role="article"]') === article && a.getClientRects().length)
      .map(a => a.href)
      .filter(href => /\/groups\/[^/]+\/(?:posts|permalink)\/\d+|multi_permalinks=\d+/.test(href));
    const excerpt = (article.innerText || '').replace(/\s+/g, ' ').slice(0, 200);
    for (const url of new Set(links)) result.push({url, labels, excerpt});
  }
  // Some Facebook feeds mark only the post header as article. The count is
  // then in a sibling footer. Climb only while the container has one post ID.
  for (const anchor of document.querySelectorAll('a[href]')) {
    const identifier = postId(anchor.href);
    if (!identifier || !anchor.getClientRects().length) continue;
    const nearestArticle = anchor.closest('[role="article"]');
    let root = nearestArticle || anchor.parentElement;
    for (let depth = 0; root && depth < 12; depth++, root = root.parentElement) {
      if (root.matches('[role="article"]') && root !== nearestArticle) break;
      const ids = new Set([...root.querySelectorAll('a[href]')].map(a => postId(a.href)).filter(Boolean));
      if (ids.size > 1) break;
      const labels = labelsIn(root, root.matches('[role="article"]'));
      if (labels.length) {
        result.push({url: anchor.href, labels, excerpt: (root.innerText || '').replace(/\s+/g, ' ').slice(0, 200)});
        break;
      }
    }
  }
  return result;
}"""


def group_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    if parsed.scheme != "https" or parsed.hostname not in ("facebook.com", "www.facebook.com"):
        raise ValueError("Group URL must be HTTPS on facebook.com")
    if parsed.username or parsed.password or parsed.port:
        raise ValueError("Group URL must not contain credentials or a port")
    if not re.fullmatch(r"/groups/[^/]+/?", parsed.path):
        raise ValueError("Expected a Facebook group URL")
    return "https://www.facebook.com" + parsed.path.rstrip("/") + "/"


def read_groups(path: Path) -> list[str]:
    values = [group_url(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
              if line.strip() and not line.lstrip().startswith("#")]
    if not values:
        raise ValueError("No authorized group URLs")
    return list(dict.fromkeys(values))


def comment_count(value: str) -> int | None:
    # Facebook's rendered count may be aria-only on the Comment action.
    # Match that specific UI format rather than numbers in arbitrary prose.
    action = re.fullmatch(r"Comment on .+'s post,\s*(.+)", value.strip(), re.I)
    if action:
        value = action[1]
    matched = COUNT.fullmatch(value)
    if not matched:
        return None
    number, unit = matched["number"], (matched["unit"] or "").lower()
    if unit:
        if number.count(",") + number.count(".") > 1:
            return None
        multiplier = 1_000_000 if unit in ("m", "triệu") else 1_000
        return int(float(number.replace(",", ".")) * multiplier)
    if number.isdigit():
        return int(number)
    # Unqualified integer counts use three-digit thousands groups.
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", number):
        return int(number.replace(",", "").replace(".", ""))
    return None


def canonical_post(value: str, allowed_groups: set[str]) -> tuple[str, str] | None:
    parsed = urlsplit(value)
    if parsed.scheme != "https" or parsed.hostname not in ("facebook.com", "www.facebook.com"):
        return None
    if parsed.username or parsed.password or parsed.port:
        return None
    match = re.fullmatch(r"/groups/([^/]+)/(?:posts|permalink)/(\d+)/?", parsed.path)
    if match:
        group, identifier = match.groups()
    else:
        group_match = re.fullmatch(r"/groups/([^/]+)(?:/search)?/?", parsed.path)
        identifier = parse_qs(parsed.query).get("multi_permalinks", [""])[0]
        if not group_match or not identifier.isdigit():
            return None
        group = group_match[1]
    if group not in allowed_groups:
        return None
    return identifier, f"https://www.facebook.com/groups/{group}/posts/{identifier}/"


def known_post_ids(annotation_dir: Path) -> set[str]:
    paths = sorted(annotation_dir.glob("thanh_vien_[1-4].csv"))
    if not paths:
        raise ValueError("No source annotations found; cannot exclude old posts safely")
    identifiers = set()
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or "post_id" not in reader.fieldnames:
                raise ValueError(f"Missing post_id: {path}")
            for row in reader:
                identifier = row["post_id"].rsplit(":", 1)[-1]
                if identifier.isdigit():
                    identifiers.add(identifier)
    if not identifiers:
        raise ValueError("No numeric old post IDs; cannot safely discover new posts")
    return identifiers


def discovery_sources(groups: list[str], keywords: list[str], search_only: bool = False) -> list[tuple[str, str]]:
    words = list(dict.fromkeys(word.strip() for word in keywords if word.strip()))
    if search_only and not words:
        raise ValueError("Search-only discovery requires at least one keyword")
    tasks = []
    for source in groups:
        source = group_url(source)
        if not search_only:
            tasks.append((source, source))
        for word in words:
            tasks.append((source, source + "search/?q=" + quote(word, safe="")))
    return tasks


def accept_candidates(raw: list[dict], groups: set[str], excluded: set[str], minimum: int) -> list[dict]:
    accepted = {}
    for item in raw:
        post = canonical_post(item["url"], groups)
        if post is None or post[0] in excluded:
            continue
        counts = [(comment_count(label), label) for label in item["labels"]]
        counts = [(count, label) for count, label in counts if count is not None and count >= minimum]
        if not counts:
            continue
        count, evidence = max(counts)
        identifier, url = post
        record = {"post_id": identifier, "url": url, "observed_comments": count,
                  "count_evidence": evidence, "post_excerpt": item.get("excerpt", "")}
        if identifier not in accepted or count > accepted[identifier]["observed_comments"]:
            accepted[identifier] = record
    return list(accepted.values())


def observe_search_post(page, allowed_groups: set[str], minimum: int, tried: set[str], *, wait_ms: int, check_access, report_path: Path) -> list[dict]:
    """Open an observed comment-count control to read that card's post URL.

    Search results initially expose timestamps pointing back to the search
    page. Opening a post is read-only; no text or reaction is submitted.
    """
    original = page.url
    if not re.fullmatch(r"/groups/[^/]+/search/?", urlsplit(original).path):
        return []
    controls = []
    for control in page.locator('[aria-label^="Comment on "]').element_handles():
        label = control.get_attribute('aria-label') or ''
        count = comment_count(label)
        if count is not None and count >= minimum and label not in tried and control.is_visible():
            controls.append((count, label, control))
    if not controls:
        return []
    _, label, control = max(controls, key=lambda item: item[0])
    tried.add(label)
    position = page.evaluate('() => ({x: scrollX, y: scrollY})')
    control.click(timeout=5000)
    page.wait_for_timeout(wait_ms)
    # On a login/checkpoint, leave the page untouched for human verification.
    check_access(page)
    urls = [page.url]
    for dialog in page.get_by_role('dialog').element_handles():
        if dialog.is_visible():
            urls.extend(dialog.eval_on_selector_all('a[href]', '(nodes) => nodes.map(n => n.href)'))
    observed = {}
    for url in urls:
        post = canonical_post(url, allowed_groups)
        if post is not None:
            observed[post[0]] = post[1]
    records = []
    if len(observed) == 1:
        records.append({'url': next(iter(observed.values())), 'labels': [label], 'excerpt': ''})
    else:
        # Keep one actual UI snapshot for an unresolved card, rather than
        # inventing a post ID or borrowing the neighboring post's count.
        page.screenshot(path=str(report_path.with_name(report_path.stem + '_unresolved.png')))
    if page.url != original:
        page.go_back(wait_until='domcontentloaded', timeout=60000)
        page.wait_for_timeout(wait_ms)
        check_access(page)
    elif page.get_by_role('dialog').count():
        page.keyboard.press('Escape')
        page.wait_for_timeout(300)
    if page.url != original:
        raise ValueError('Search post did not return to the same source page')
    page.evaluate('(p) => scrollTo(p.x, p.y)', position)
    return records


def run_discovery(page, *, groups: list[str], annotation_dir: Path, output_urls: Path,
                  report_path: Path, rounds: int, minimum: int, max_posts: int,
                  wait_ms: int, check_access, keywords: list[str] | None = None,
                  search_only: bool = False) -> dict:
    old_ids = known_post_ids(annotation_dir)
    accepted = {}
    scanned = []
    visited = []
    diagnostics = []
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_urls.parent.mkdir(parents=True, exist_ok=True)

    def save():
        ranked = sorted(accepted.values(), key=lambda row: (-row["observed_comments"], row["post_id"]))
        with report_path.with_suffix(".tmp").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(ranked)
        report_path.with_suffix(".tmp").replace(report_path)
        output_urls.with_suffix(".tmp").write_text(
            "# Observed new posts only; minimum observed comments: " + str(minimum) + "\n" +
            "".join(row["url"] + "\n" for row in ranked), encoding="utf-8",
        )
        output_urls.with_suffix(".tmp").replace(output_urls)
        status = {"accepted_posts": len(ranked), "excluded_old_post_ids": len(old_ids),
                  "minimum_comments": minimum, "scanned_groups": scanned,
                  "visited_sources": visited,
                  "urls_file": str(output_urls), "candidate_report": str(report_path),
                  "diagnostics": diagnostics[-12:]}
        report_path.with_suffix(".json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        return status

    if output_urls.exists():
        raise ValueError("Discovery output already exists; choose another --urls to preserve previous findings")
    for source, browse_url in discovery_sources(groups, keywords or [], search_only):
        print(f"Tìm link trong nhóm: {browse_url}", flush=True)
        page.goto(browse_url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(wait_ms)
        check_access(page)
        allowed = {urlsplit(source).path.strip("/").split("/")[1]}
        current = urlsplit(page.url)
        match = re.match(r"/groups/([^/]+)", current.path)
        if current.hostname in ("facebook.com", "www.facebook.com") and match:
            allowed.add(match[1])
        else:
            raise ValueError("Group navigation did not resolve to a Facebook group")
        if source not in scanned:
            scanned.append(source)
        visited.append(browse_url)
        idle = 0
        last_seen = set()
        tried_search_cards = set()
        for number in range(1, rounds + 1):
            check_access(page)
            raw = page.evaluate(EXTRACT)
            if '/search/' in browse_url:
                raw.extend(observe_search_post(page, allowed, minimum, tried_search_cards, wait_ms=wait_ms,
                                               check_access=check_access, report_path=report_path))
            if number == 1 or (number % 5 == 0 and not accepted):
                if number == 1 and not raw:
                    page.screenshot(path=str(report_path.with_suffix('.png')))
                probe = page.evaluate(r"""() => ({
                  article_count: document.querySelectorAll('[role="article"]').length,
                  group_link_samples: [...new Set([...document.querySelectorAll('a[href]')].map(n => {
                    try {
                      const u = new URL(n.href);
                      if (!/\/groups\//.test(u.pathname) || /\/user\//.test(u.pathname)) return '';
                      const query = new URLSearchParams();
                      for (const key of ['multi_permalinks','story_fbid']) if (u.searchParams.has(key)) query.set(key, u.searchParams.get(key));
                      return u.origin + u.pathname + (query.size ? '?' + query.toString() : '');
                    } catch { return ''; }
                  }).filter(Boolean))].slice(0, 30),
                  page_counter_lines: (document.body.innerText || '').split('\n').filter(s => /bình luận|comments?/i.test(s)).slice(0, 20),
                  page_counter_labels: [...document.querySelectorAll('[aria-label]')].map(n => n.getAttribute('aria-label')).filter(s => /\d.*(?:bình luận|comments?)/i.test(s || '')).slice(0, 12),
                  articles: [...document.querySelectorAll('[role="article"]')].filter(a => a.getClientRects().length).slice(0, 4).map(a => ({
                    links: [...a.querySelectorAll('a[href]')].map(n => {
                      try { const u = new URL(n.href); return u.origin + u.pathname; } catch { return ''; }
                    }).filter(s => /\/groups\/[^/]+\/(?:posts|permalink)\//.test(s)).slice(0, 12),
                    counter_lines: (a.innerText || '').split('\n').filter(s => /bình luận|comments?/i.test(s)).slice(0, 8),
                    counter_labels: [...a.querySelectorAll('[aria-label]')].map(n => n.getAttribute('aria-label')).filter(s => /\d.*(?:bình luận|comments?)/i.test(s || '')).slice(0, 8)
                  }))
                })""")
                raw_posts = [{"url": item['url'], "excluded_old_id": bool((post := canonical_post(item['url'], allowed)) and post[0] in old_ids)} for item in raw]
                diagnostics.append({"group_url": source, "round": number, "raw_candidates": len(raw), "raw_posts": raw_posts, **probe})
            fingerprint = {entry["url"] for entry in raw}
            idle = idle + 1 if fingerprint == last_seen else 0
            last_seen = fingerprint
            found = accept_candidates(raw, allowed, old_ids, minimum)
            for record in found:
                identifier = record["post_id"]
                record["group_url"] = source
                if identifier not in accepted and len(accepted) >= max_posts:
                    continue
                if identifier not in accepted or record["observed_comments"] > accepted[identifier]["observed_comments"]:
                    accepted[identifier] = record
            save()
            print(f"  Lượt {number}/{rounds}: tổng {len(accepted)} bài mới có >= {minimum} bình luận.", flush=True)
            if len(accepted) >= max_posts:
                return save()
            if idle >= 8:
                print("  Không có thêm link bài; chuyển nhóm tiếp theo.", flush=True)
                break
            page.mouse.wheel(0, 1800)
            page.wait_for_timeout(wait_ms)
    return save()
