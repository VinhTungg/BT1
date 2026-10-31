# -*- coding: utf-8 -*-
"""Chia dữ liệu train/val/test chống data leakage theo nhóm bài viết.

Toàn bộ câu thuộc cùng một ``post_id`` luôn nằm trong cùng một tập
(ràng buộc bắt buộc chống data leakage).

Phương pháp splitting: thuật toán chia nhóm tự thiết kế có xét cân bằng
nhãn — thử nhiều candidate hoán vị nhóm (sinh từ ``random.Random(seed + i)``),
gán nhóm bằng chiến lược largest-remaining-bin, chấm điểm theo lệch tỷ lệ
mẫu và lệch phân bố nhãn, chọn candidate tốt nhất. Mọi bước deterministic
với seed cố định và tie-break theo thứ tự candidate.

Chạy từ repository root:

    python scripts/make_splits.py
    python scripts/make_splits.py --input data/processed/cleaned.csv \
        --output-dir data/splits --report-dir reports --seed 42
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
N_CANDIDATES = 200

BIN_ORDER = ("train", "val", "test")
BIN_TARGET_RATIO = {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO}

SPLIT_COLUMN_ORDER = [
    "id",
    "post_id",
    "text",
    "text_clean",
    "text_tokenized",
    "label",
    "source_file",
    "source_row",
    "source_url",
    "collected_at",
    "extraction_method",
]

logger = logging.getLogger("make_splits")


# ---------------------------------------------------------------------------
# Chuẩn bị dữ liệu & nhóm
# ---------------------------------------------------------------------------
def load_input(input_csv: Path) -> pd.DataFrame:
    """Đọc dữ liệu đã clean, xác thực schema và ràng buộc bắt buộc.

    Raises:
        ValueError: nếu thiếu cột bắt buộc.
        RuntimeError: nếu còn bản ghi thiếu post_id (không thể bảo toàn ràng
            buộc chống leakage) hoặc label không thuộc {0,1}.
    """
    if not input_csv.exists():
        raise FileNotFoundError(f"Không tìm thấy file đầu vào: {input_csv}")
    df = pd.read_csv(input_csv, encoding="utf-8", dtype=str, keep_default_na=False)
    required = {"id", "post_id", "text", "text_clean", "text_tokenized", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"File đầu vào thiếu cột bắt buộc: {sorted(missing)}")

    empty_post = df["post_id"].str.strip() == ""
    if empty_post.any():
        raise RuntimeError(
            f"{int(empty_post.sum())} bản ghi thiếu post_id; không thể đảm bảo "
            "ràng buộc chống leakage theo nhóm. Cần khôi phục post_id từ dữ liệu "
            "nguồn/metadata trước khi chia (không tự sinh post_id ngẫu nhiên)."
        )

    bad_label = ~df["label"].isin(["0", "1"])
    if bad_label.any():
        raise RuntimeError(
            f"{int(bad_label.sum())} bản ghi label không thuộc {{0,1}}; chạy lại "
            "scripts/audit_data.py trước khi chia."
        )
    df["label"] = df["label"].astype(int)
    return df


def build_posts(df: pd.DataFrame) -> pd.DataFrame:
    """Tổng hợp thống kê từng nhóm (post_id): số câu, số toxic."""
    posts = (
        df.groupby("post_id", as_index=False)
        .agg(n=("id", "size"), toxic=("label", "sum"))
        .sort_values("post_id", kind="stable")
        .reset_index(drop=True)
    )
    posts["toxic"] = posts["toxic"].astype(int)
    return posts


# ---------------------------------------------------------------------------
# Thuật toán chia nhóm
# ---------------------------------------------------------------------------
def split_groups_by_capacity(
    order: list[str], sizes: dict[str, int], targets: dict[str, float]
) -> dict[str, list[str]]:
    """Gán nhóm post theo ``order`` vào bin còn thiếu năng lực lớn nhất.

    Mỗi bài viết được gán nguyên vẹn vào bin train/val/test đang thiếu
    nhiều nhất so với target (gán tiếp trong bin đó đã vượt target cũng chỉ
    lệch nhỏ). Tie-break theo thứ tự bin (train, val, test) → deterministic.
    """
    bins: dict[str, list[str]] = {name: [] for name in BIN_ORDER}
    committed: dict[str, int] = {name: 0 for name in BIN_ORDER}
    remaining = list(order)
    # Seed: với >= 3 nhóm, gán nhóm đầu tiên theo thứ tự vào từng bin để
    # không bin nào rỗng (tránh cấu trúc split lệch vô nghĩa).
    if len(remaining) >= len(BIN_ORDER):
        for name in BIN_ORDER:
            post_id = remaining.pop(0)
            bins[name].append(post_id)
            committed[name] += sizes[post_id]
    for post_id in remaining:
        best_bin = BIN_ORDER[0]
        best_gap = targets[BIN_ORDER[0]] - committed[BIN_ORDER[0]]
        for name in BIN_ORDER[1:]:
            gap = targets[name] - committed[name]
            if gap > best_gap:
                best_gap = gap
                best_bin = name
        bins[best_bin].append(post_id)
        committed[best_bin] += sizes[post_id]
    return bins


def evaluate_candidate(
    bins: dict[str, list[str]], post_stats: dict[str, tuple[int, int]], n_samples: int
) -> float:
    """Cost của một candidate: sai lệch tỷ lệ + lệch cân bằng nhãn.

    post_stats: post_id -> (n_samples, n_toxic) của nhóm.
    Penalize tập nào thiếu hẳn một lớp (toàn dataset có cả hai lớp).
    """
    toxic_total = sum(t for _, t in post_stats.values())
    toxic_overall = toxic_total / n_samples if n_samples else 0.0

    ratio_dev = 0.0
    toxic_dev = 0.0
    class_penalty = 0.0
    for name, group in bins.items():
        target_ratio = BIN_TARGET_RATIO[name]
        n_split = sum(post_stats[p][0] for p in group)
        t_split = sum(post_stats[p][1] for p in group)
        ratio_dev += abs(n_split / n_samples - target_ratio)
        if n_split > 0:
            toxic_dev += abs(t_split / n_split - toxic_overall)
            if toxic_total > 0 and t_split == 0:
                class_penalty += 10.0
            if 0 < toxic_total < n_samples and t_split == n_split:
                class_penalty += 10.0
        else:
            class_penalty += 10.0  # bin rỗng là lỗi cấu trúc nghiêm trọng
    return 2.0 * ratio_dev + 5.0 * toxic_dev + class_penalty


def find_best_split(
    posts: pd.DataFrame, seed: int, n_candidates: int = N_CANDIDATES
) -> dict[str, list[str]]:
    """Chia nhóm post_id thành train/val/test, chọn candidate cost thấp nhất.

    Candidate thứ ``i`` dùng ``random.Random(seed + i)``; mỗi candidate là
    một hoán vị nhóm được gán bằng largest-remaining-bin. Số candidate giới
    hạn hữu hạn; tie-break (cost nhỏ hơn mới đổi best) theo thứ tự candidate
    → kết quả reproducible với cùng seed.
    """
    n_samples = int(posts["n"].sum())
    sizes = {str(p): int(n) for p, n in zip(posts["post_id"], posts["n"])}
    toxic = {str(p): int(t) for p, t in zip(posts["post_id"], posts["toxic"])}
    post_stats = {p: (sizes[p], toxic[p]) for p in sizes}
    targets = {
        "train": TRAIN_RATIO * n_samples,
        "val": VAL_RATIO * n_samples,
        "test": TEST_RATIO * n_samples,
    }
    # Danh sách nhóm được sắp theo post_id để thứ tự danh sách ổn định.
    post_ids = sorted(sizes)
    if len(post_ids) < 3:
        raise RuntimeError(
            f"Chỉ có {len(post_ids)} nhóm post_id; không thể chia 3 train/val/test."
        )

    best_cost = float("inf")
    best_bins: dict[str, list[str]] | None = None
    for i in range(n_candidates):
        rng = random.Random(seed + i)
        order = list(post_ids)
        rng.shuffle(order)
        bins = split_groups_by_capacity(order, sizes, targets)
        cost = evaluate_candidate(bins, post_stats, n_samples)
        if cost < best_cost:
            best_cost = cost
            best_bins = bins
    if best_bins is None:
        raise RuntimeError("Không sinh được split hợp lệ từ các nhóm đã cho.")
    return best_bins


# ---------------------------------------------------------------------------
# Kiểm tra leakage & bảo toàn dữ liệu
# ---------------------------------------------------------------------------
def verify_no_group_leakage(splits: dict[str, pd.DataFrame]) -> dict[str, bool]:
    """Kiểm chồng lấn post_id giữa các tập; mọi giá trị phải True."""
    post_sets = {name: set(s["post_id"]) for name, s in splits.items()}
    return {
        "train_val": post_sets["train"].isdisjoint(post_sets["val"]),
        "train_test": post_sets["train"].isdisjoint(post_sets["test"]),
        "val_test": post_sets["val"].isdisjoint(post_sets["test"]),
    }


def verify_record_conservation(
    before: pd.DataFrame, splits: dict[str, pd.DataFrame]
) -> dict[str, bool]:
    """Xác nhận không mất/sinh bản ghi, multiset `id` bảo toàn 1-1."""
    before_ids = sorted(before["id"])
    split_frames = list(splits.values())
    after_ids = sorted(itertools.chain.from_iterable(s["id"] for s in split_frames))
    all_ids = [i for s in split_frames for i in s["id"]]
    return {
        "total_conserved": sum(len(s) for s in split_frames) == len(before),
        "id_multiset_equal": before_ids == after_ids,
        "no_duplicate_ids_in_splits": len(all_ids) == len(set(all_ids)),
    }


def find_cross_split_text_duplicates(
    splits: dict[str, pd.DataFrame]
) -> list[dict[str, str]]:
    """Exact duplicate text (theo `text_clean`, chuẩn hóa lower) giữa các tập.

    Chuẩn hóa chỉ phục vụ kiểm tra; không thay đổi text gốc. Câu ngắn phổ
    biến trùng tự nhiên được coi là nghi vấn cần nhóm review, không tự sửa.
    """
    normalized: dict[str, dict[str, str]] = {}
    for name, s in splits.items():
        normalized[name] = dict(
            zip(s["id"], s["text_clean"].astype(str).str.lower().str.strip())
        )
    reports: list[dict[str, str]] = []
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        left_texts = set(normalized[left].values())
        for rid, text in normalized[right].items():
            if text and text in left_texts:
                reports.append({
                    "pair": f"{left}/{right}",
                    "right_id": rid,
                    "detail": "Cùng văn bản xuất hiện ở hai tập; cần nhóm kiểm tra.",
                })
    return reports


# ---------------------------------------------------------------------------
# Cắt DataFrame & ghi output
# ---------------------------------------------------------------------------
def make_split_frames(
    df: pd.DataFrame, bins: dict[str, list[str]]
) -> dict[str, pd.DataFrame]:
    """Cắt df theo nhóm đã chia, ordering ổn định theo (post_id, id)."""
    splits: dict[str, pd.DataFrame] = {}
    for name, group in bins.items():
        part = df[df["post_id"].isin(group)].copy()
        part["label"] = part["label"].astype(int)
        splits[name] = part.sort_values(["post_id", "id"], kind="stable").reset_index(drop=True)
    return splits


def render_split_report(
    df: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    posts: pd.DataFrame,
    stats: dict[str, Any],
    method_note: str,
) -> str:
    """Sinh nội dung markdown `split_report.md` từ kết quả thực tế."""
    n_total = len(df)
    toxic_total = int((df["label"] == 1).sum())
    ratio_labels = {"train": "70%", "val": "15%", "test": "15%"}
    lines = [
        "# Báo cáo Split dữ liệu",
        "",
        "## Phương pháp splitting",
        "",
        "- **Group-aware splitting theo `post_id`**: mỗi nhóm (các câu cùng bài",
        "  viết) được gán nguyên vẹn vào một tập duy nhất, chống data leakage.",
        "- Thuật toán: thử giới hạn hữu hạn candidate hoán vị nhóm (mỗi candidate",
        "  dùng `random.Random(seed + i)`), gán nhóm bằng chiến lược",
        "  *largest-remaining-bin* theo tỷ lệ 70/15/15, chấm điểm cost theo:",
        "  (1) sai lệch tỷ lệ số mẫu so với mục tiêu;",
        "  (2) sai lệch tỷ lệ toxic giữa tập với toàn dataset (trọng số 5);",
        "  (3) hình phạt lớn nếu tập nào thiếu hẳn một lớp.",
        "- Chọn candidate có cost thấp nhất; tie-break theo thứ tự candidate →",
        "  kết quả deterministic, reproducible với cùng seed.",
        f"- Không dùng `StratifiedGroupKFold` trực tiếp vì số nhóm rất nhỏ "
        f"({posts.shape[0]} post) và fold-based split không bỏ kiểm soát được tỷ lệ "
        "70/15/15; thuật toán trên kiểm soát cả tỷ lệ, nhóm và cân bằng nhãn.",
        "",
        f"- Random seed: `{stats['seed']}`",
        f"- Số candidate thử: `{stats['n_candidates']}`",
        f"- {method_note}",
        "",
        "## Tổng quan dữ liệu",
        "",
        f"- Tổng số samples: {n_total}",
        f"- Tổng số unique posts: {posts.shape[0]}",
        f"- Tổng số toxic: {toxic_total} ({100 * toxic_total / n_total:.2f}%)",
        f"- Size nhóm (câu mỗi post): min {posts['n'].min()}, "
        f"median {posts['n'].median():.0f}, max {posts['n'].max()}, "
        f"mean {posts['n'].mean():.2f}",
        "",
        "## Thống kê từng tập",
        "",
        "| Split | Samples | Posts | Non-toxic | Toxic | Toxic % | Tỷ lệ mẫu vs mục tiêu |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name in BIN_ORDER:
        s = splits[name]
        n = len(s)
        t = int((s["label"] == 1).sum())
        toxic_pct = 100 * t / n if n else 0.0  # bin rỗng chỉ xảy ra với dữ liệu bị lỗi
        lines.append(
            f"| {name} | {n} | {s['post_id'].nunique()} | {n - t} | {t} |"
            f" {toxic_pct:.2f}% | {100 * n / n_total:.2f}% (mục tiêu {ratio_labels[name]}) |"
        )
    checks = stats["group_leakage"]
    record_checks = stats["record_checks"]
    lines += [
        "",
        "## Kiểm tra chống leakage & bảo toàn dữ liệu",
        "",
        f"- Train/Val post overlap: {'PASS' if checks['train_val'] else 'FAIL'}",
        f"- Train/Test post overlap: {'PASS' if checks['train_test'] else 'FAIL'}",
        f"- Val/Test post overlap: {'PASS' if checks['val_test'] else 'FAIL'}",
        f"- Tổng số bản ghi các tập = đầu vào: "
        f"{'PASS' if record_checks['total_conserved'] else 'FAIL'}",
        f"- Multiset `id` bảo toàn (không mất/sinh bản ghi): "
        f"{'PASS' if record_checks['id_multiset_equal'] else 'FAIL'}",
        f"- Không có `id` trùng trong split outputs: "
        f"{'PASS' if record_checks['no_duplicate_ids_in_splits'] else 'FAIL'}",
        "",
        "## Duplicate text bất thường giữa các tập",
        "",
    ]
    dupes = stats["cross_split_text_duplicates"]
    if not dupes:
        lines.append("- Không phát hiện exact duplicate text giữa train/val/test.")
    else:
        lines.append(f"- Phát hiện {len(dupes)} bản nghi vấn (cần nhóm review):")
        for entry in dupes:
            lines.append(f"  - {entry['pair']}: id `{entry['right_id']}`")
    lines += [
        "",
        "## Giới hạn còn tồn tại",
        "",
        "- Số nhóm rất nhỏ (17 post); tỷ lệ 70/15/15 chỉ đạt gần (không chính xác)",
        "  vì mỗi post có nhiều câu; ràng buộc group-aware được ưu tiên tuyệt đối.",
        "- Dataset mất cân bằng nặng (toxic chỉ ~8%); split giữ phân bố nhãn gần",
        "  tổng thể, không oversample/undersample hay tăng dữ liệu.",
        "- Duplicate check thiết kế exact (lower-case `text_clean`), không fuzzy;",
        "  câu ngắn phổ biến trùng tự nhiên cần review thủ công nếu xuất hiện.",
        "- Test set chỉ dùng cho đánh giá cuối; không dùng để fit bất kỳ",
        "  thành phần preprocessing có học tham số nào.",
    ]
    return "\n".join(lines) + "\n"


def run_split_pipeline(
    input_csv: Path,
    output_dir: Path,
    report_dir: Path,
    seed: int = RANDOM_STATE,
    n_candidates: int = N_CANDIDATES,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Pipeline chia dữ liệu hoàn chỉnh: load → chia → verify → ghi output."""
    df = load_input(input_csv)
    posts = build_posts(df)
    bins = find_best_split(posts, seed, n_candidates)
    splits = make_split_frames(df, bins)

    group_leakage = verify_no_group_leakage(splits)
    if not all(group_leakage.values()):
        failed = [k for k, ok in group_leakage.items() if not ok]
        raise RuntimeError(f"Group leakage FAILURE: {failed}; không ghi splits.")
    record_checks = verify_record_conservation(df, splits)
    if not all(record_checks.values()):
        raise RuntimeError(f"Record conservation FAILURE: {record_checks}.")

    cross_dupes = find_cross_split_text_duplicates(splits)
    stats = {
        "seed": seed,
        "n_candidates": n_candidates,
        "group_leakage": group_leakage,
        "record_checks": record_checks,
        "cross_split_text_duplicates": cross_dupes,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        extra_cols = [c for c in part.columns if c not in SPLIT_COLUMN_ORDER]
        cols = [c for c in SPLIT_COLUMN_ORDER if c in part.columns] + extra_cols
        # Ghi file tạm rồi atomic-swap để chạy lại không tạo output nửa vời.
        tmp_path = output_dir / f"{name}.csv.tmp"
        part[cols].to_csv(tmp_path, index=False, encoding="utf-8")
        tmp_path.replace(output_dir / f"{name}.csv")

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "split_report.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "n_candidates": n_candidates,
                "group_leakage": group_leakage,
                "record_checks": record_checks,
                "cross_split_text_duplicates": cross_dupes,
                "splits": {
                    name: {
                        "samples": len(p),
                        "posts": int(p["post_id"].nunique()),
                        "toxic": int((p["label"] == 1).sum()),
                        "non_toxic": int((p["label"] == 0).sum()),
                        "sample_ratio_pct": round(100 * len(p) / len(df), 2),
                    }
                    for name, p in splits.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (report_dir / "split_report.md").write_text(
        render_split_report(
            df, splits, posts, stats,
            method_note=(f"Input: {input_csv.as_posix()}, output: "
                         f"{output_dir.as_posix()}/{{train,val,test}}.csv."),
        ),
        encoding="utf-8",
    )
    logger.info(
        "Split xong: train %d / val %d / test %d (seed=%d)",
        len(splits["train"]), len(splits["val"]), len(splits["test"]), seed,
    )
    return splits, stats


def build_arg_parser() -> argparse.ArgumentParser:
    """Xây dựng parser CLI."""
    parser = argparse.ArgumentParser(
        description="Group-aware train/val/test split chống data leakage."
    )
    parser.add_argument(
        "--input", type=Path,
        default=REPO_ROOT / "data" / "processed" / "cleaned.csv",
        help="File CSV đầu vào đã tiền xử lý.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "data" / "splits",
        help="Thư mục ghi train.csv/val.csv/test.csv.",
    )
    parser.add_argument(
        "--report-dir", type=Path, default=REPO_ROOT / "reports",
        help="Thư mục chứa báo cáo split.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--n-candidates", type=int, default=N_CANDIDATES)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    run_split_pipeline(
        args.input, args.output_dir, args.report_dir,
        seed=args.seed, n_candidates=args.n_candidates,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
