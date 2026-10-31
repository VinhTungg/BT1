# -*- coding: utf-8 -*-
"""Gộp và kiểm tra chất lượng dữ liệu annotation Toxic/Non-toxic tiếng Việt.

Đọc tất cả file CSV annotation hợp lệ trong thư mục đầu vào, chuẩn hóa schema,
kiểm tra giá trị thiếu, label không hợp lệ, duplicate và thống kê tổng thể,
rồi ghi ra `data/processed/merged.csv` cùng các báo cáo trong `reports/`.

Script không bao giờ sửa hoặc ghi đè file annotation gốc. Nhãn không được
tự ý thay đổi; bản ghi lỗi chỉ bị loại khi có log truy vết rõ ràng.

Chạy từ repository root:

    python scripts/audit_data.py
    python scripts/audit_data.py --input-dir annotations \
        --output-csv data/processed/merged.csv --report-dir reports
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

# Cột bắt buộc phải có trong mọi file annotation.
REQUIRED_COLUMNS = ("id", "post_id", "text", "label")
# Thứ tự cột chuẩn của bản ghi gộp (các cột phụ hợp lệ được nối phía sau).
MERGED_COLUMN_ORDER = [
    "id",
    "post_id",
    "text",
    "label",
    "source_file",
    "source_row",
    "source_url",
    "collected_at",
    "extraction_method",
    "note",
]

logger = logging.getLogger("audit_data")


# ---------------------------------------------------------------------------
# Nạp dữ liệu
# ---------------------------------------------------------------------------
def load_annotation_file(path: Path) -> pd.DataFrame:
    """Đọc một file annotation CSV, trả về DataFrame dạng chuỗi.

    Sử dụng ``utf-8-sig`` để hỗ trợ UTF-8 có/không BOM. Mọi ô được đọc là
    chuỗi với ``keep_default_na=False`` nên không sinh ``NaN`` và không biến
    giá trị trống thành "nan"/"None". Các cột dồn có tên rỗng (khoảng trắng
    cuối) bị loại vì không mang dữ liệu.

    Raises:
        ValueError: nếu CSV sai định dạng hoặc thiếu cột bắt buộc.
    """
    try:
        df = pd.read_csv(
            path,
            encoding="utf-8-sig",
            dtype=str,
            keep_default_na=False,
            na_values=[],
            skip_blank_lines=True,
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise ValueError(f"Không đọc được CSV {path.name}: {exc}") from exc

    # Loại các cột không mang dữ liệu: tên rỗng/phân tách thừa (pandas
    # đặt tên "Unnamed: N"). Các cột có tên nhưng rỗng (ví dụ `note`) được
    # giữ lại để schema thống nhất giữa các file khi concat.
    named_cols = [
        c for c in df.columns
        if str(c).strip() != "" and not str(c).startswith("Unnamed:")
    ]
    df = df.loc[:, named_cols]
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"File {path.name} thiếu cột bắt buộc: {missing}. "
            f"Các cột hiện có: {list(df.columns)}"
        )
    return df


def parse_label(raw: str | float | None) -> int | None:
    """Chuyển biểu diễn label thô thành ``0``/``1``; không rõ nghĩa trả None.

    Chỉ chấp nhận "0", "1" sau khi bỏ khoảng trắng (ví dụ "0 ", " 1").
    Giá trị trống, "nan", "None" hoặc bất kỳ dạng nào khác đều trả về None
    (không tự suy đoán sang nhãn 0/1).
    """
    if raw is None:
        return None
    stripped = str(raw).strip()
    if stripped == "0":
        return 0
    if stripped == "1":
        return 1
    return None


def prepare_annotation_frame(path: Path) -> pd.DataFrame:
    """Đọc một file và gắn cột truy vết ``source_file``, ``source_row``."""
    df = load_annotation_file(path)
    out = df.copy()
    out["source_file"] = path.name
    # source_row: số thứ tự dòng dữ liệu (1 = dòng dữ liệu đầu tiên).
    out["source_row"] = range(1, len(out) + 1)
    return out


def load_all_annotations(input_dir: Path) -> dict[str, Any]:
    """Đọc mọi file ``*.csv`` trong ``input_dir``, sắp theo tên file.

    Returns:
        dict gồm ``frames`` (danh sách DataFrame đọc thành công), ``errors``
        ((filename, message) cho các file lỗi định dạng/schema) và
        ``files_found`` (số file CSV tìm thấy).
    """
    csv_paths = sorted(p for p in input_dir.glob("*.csv") if p.is_file())
    frames: list[pd.DataFrame] = []
    errors: list[tuple[str, str]] = []
    for path in csv_paths:
        try:
            frame = prepare_annotation_frame(path)
            frames.append(frame)
            logger.info("Đọc OK: %s (%d dòng)", path.name, len(frame))
        except ValueError as exc:
            logger.error("%s", exc)
            errors.append((path.name, str(exc)))
    return {"frames": frames, "errors": errors, "files_found": len(csv_paths)}


# ---------------------------------------------------------------------------
# Kiểm tra chất lượng
# ---------------------------------------------------------------------------
def tag_invalid_records(df: pd.DataFrame) -> pd.DataFrame:
    """Gắn cột ``issue`` and ``_parsed_label`` cho từng bản ghi.

    Phát hiện: label trống/không thuộc {0,1}, text trống hoặc chỉ khoảng
    trắng, post_id trống, id trống. Bản ghi hợp lệ có ``issue == ""``.
    """
    parsed_labels: list[int | None] = [parse_label(v) for v in df["label"]]
    issues: list[str] = []
    for parsed, raw_label, text, post_id, record_id in zip(
        parsed_labels, df["label"], df["text"], df["post_id"], df["id"]
    ):
        parts: list[str] = []
        if parsed is None:
            parts.append("invalid_or_missing_label")
        if not str(text).strip():
            parts.append("empty_or_whitespace_text")
        if not str(post_id).strip():
            parts.append("empty_post_id")
        if not str(record_id).strip():
            parts.append("empty_id")
        issues.append(";".join(parts))

    out = df.copy()
    out["_parsed_label"] = parsed_labels
    out["issue"] = issues
    return out


def find_invalid(tagged: pd.DataFrame) -> pd.DataFrame:
    """Trả về các bản ghi bị tag có vấn đề ở :func:`tag_invalid_records`."""
    return tagged[tagged["issue"] != ""].copy()


def find_duplicates(valid: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Phát hiện và xử lý duplicate trên bản ghi hợp lệ.

    Chính sách (không âm thầm giảm dữ liệu):
      * Trùng ``id`` (trong file hoặc giữa các file) → giữ bản ghi xuất hiện
        đầu tiên, ghi log đầy đủ cho mỗi bản ghi bị loại.
      * Trùng toàn bộ bản ghi (trường hợp id đọc khác do lỗi thu thập) →
        tương tự.
      * Cùng văn bản nhưng khác ``id`` → KHÔNG loại, chỉ báo cáo nghi vấn.
      * Cùng văn bản nhưng nhãn khác nhau → conflicting annotation: KHÔNG
        loại, KHÔNG chọn majority vote, chỉ báo cáo.
      * Nhiều câu cùng ``post_id`` → hành vi hợp lệ, chỉ thống kê.

    Returns:
        (DataFrame đã loại duplicate an toàn, danh sách bản ghi issue/log)
    """
    removed_log: list[dict[str, Any]] = []
    keep = valid.copy()

    # 1-2. Trùng id (trong cùng file hoặc giữa các file).
    dup_id_mask = keep.duplicated(subset="id", keep="first")
    for _, row in keep[dup_id_mask].iterrows():
        removed_log.append(_duplicate_log_row(row, "duplicate_id"))
    keep = keep[~dup_id_mask]

    # 3. Trùng hoàn toàn toàn bộ bản ghi (các cột dữ liệu, bỏ cột truy vết).
    data_cols = [
        c for c in keep.columns
        if c not in ("source_file", "source_row", "_parsed_label")
    ]
    full_dup_mask = keep.duplicated(subset=data_cols, keep="first")
    for _, row in keep[full_dup_mask].iterrows():
        removed_log.append(_duplicate_log_row(row, "duplicate_full_record"))
    keep = keep[~full_dup_mask]

    # 6. (thống kê) Nhiều câu cùng post_id chỉ được đếm, không phải lỗi:
    #    số post_id được tính trong compute_statistics.

    conflicting: list[dict[str, Any]] = []
    suspicious: list[dict[str, Any]] = []
    if not keep.empty:
        # 5. Cùng văn bản, nhãn khác nhau — báo cáo, không sửa nhãn.
        label_variants = keep.groupby("text")["label"].nunique()
        conflict_texts = set(label_variants[label_variants > 1].index)
        for _, row in keep[keep["text"].isin(conflict_texts)].iterrows():
            conflicting.append(_report_row(
                row, "conflicting_label_same_text",
                "Cùng văn bản được gán nhãn khác nhau; giữ nguyên, cần nhóm duyệt lại.",
            ))

        # 4. Trùng văn bản (khác id) — báo cáo nghi vấn, không loại.
        text_counts = keep.groupby("text").size()
        dup_texts = set(text_counts[text_counts > 1].index) - conflict_texts
        for _, row in keep[keep["text"].isin(dup_texts)].iterrows():
            suspicious.append(_report_row(
                row, "duplicate_text_different_id",
                "Văn bản trùng ở các bản ghi khác nhau; giữ nguyên để nhóm kiểm tra.",
            ))

    return keep, removed_log + conflicting + suspicious


def _duplicate_log_row(row: pd.Series, reason: str) -> dict[str, Any]:
    """Tạo entry log cho một bản ghi duplicate bị loại khỏi merged."""
    return {
        "source_file": row.get("source_file", ""),
        "source_row": int(row.get("source_row", 0)),
        "id": row.get("id", ""),
        "post_id": row.get("post_id", ""),
        "issue": reason,
        "detail": "Bản ghi bị loại vì duplicate; giữ bản ghi xuất hiện đầu tiên.",
    }


def _report_row(row: pd.Series, issue: str, detail: str) -> dict[str, Any]:
    """Tạo entry báo cáo cho bản ghi nghi vấn (không bị loại)."""
    return {
        "source_file": row.get("source_file", ""),
        "source_row": int(row.get("source_row", 0)),
        "id": row.get("id", ""),
        "post_id": row.get("post_id", ""),
        "issue": issue,
        "detail": detail,
    }


# ---------------------------------------------------------------------------
# Thống kê
# ---------------------------------------------------------------------------
def compute_statistics(
    per_file_counts: dict[str, int],
    total_records: int,
    invalid: pd.DataFrame,
    valid: pd.DataFrame,
    issue_log: list[dict[str, Any]],
) -> dict[str, Any]:
    """Gom toàn bộ thống kê audit từ số liệu thực tế."""
    label_counts = valid["label"].value_counts().sort_index()
    n_valid = len(valid)
    post_counts = valid.groupby("post_id").size()
    char_len = valid["text"].str.strip().str.len()
    word_len = valid["text"].str.strip().str.split().str.len()
    toxic_ratio = (label_counts.get(1, 0) / n_valid * 100) if n_valid else 0.0
    removed_count = sum(
        1 for e in issue_log if str(e["issue"]).startswith("duplicate")
    )

    return {
        "input_files": per_file_counts,
        "total_records_before_processing": total_records,
        "valid_records": n_valid,
        "invalid_records": len(invalid),
        "removed_duplicates": removed_count,
        "conflicting_label_records": sum(
            1 for e in issue_log if e["issue"] == "conflicting_label_same_text"
        ),
        "duplicate_text_records": sum(
            1 for e in issue_log if e["issue"] == "duplicate_text_different_id"
        ),
        "label_0": int(label_counts.get(0, 0)),
        "label_1": int(label_counts.get(1, 0)),
        "label_0_ratio_pct": round(100 - toxic_ratio, 2) if n_valid else 0.0,
        "label_1_ratio_pct": round(toxic_ratio, 2) if n_valid else 0.0,
        "unique_post_ids": int(valid["post_id"].nunique()),
        "avg_sentences_per_post": round(float(post_counts.mean()), 2) if len(post_counts) else 0.0,
        "text_char_length": {
            "mean": round(float(char_len.mean()), 2) if n_valid else 0.0,
            "median": float(char_len.median()) if n_valid else 0.0,
            "min": int(char_len.min()) if n_valid else 0,
            "max": int(char_len.max()) if n_valid else 0,
        },
        "text_word_length": {
            "mean": round(float(word_len.mean()), 2) if n_valid else 0.0,
            "median": float(word_len.median()) if n_valid else 0.0,
            "min": int(word_len.min()) if n_valid else 0,
            "max": int(word_len.max()) if n_valid else 0,
        },
    }


# ---------------------------------------------------------------------------
# Báo cáo
# ---------------------------------------------------------------------------
def build_issue_rows(
    invalid: pd.DataFrame, issue_log: list[dict[str, Any]]
) -> pd.DataFrame:
    """Ghép bản ghi lỗi và log duplicate/suspicious thành bảng issue."""
    rows: list[dict[str, Any]] = []
    for _, row in invalid.iterrows():
        rows.append({
            "source_file": row["source_file"],
            "source_row": int(row["source_row"]),
            "id": row["id"],
            "post_id": row["post_id"],
            "issue": row["issue"],
            "detail": f"Bản ghi lỗi: giá trị label đọc được = {row['label']!r}.",
        })
    rows.extend(issue_log)
    return pd.DataFrame(
        rows, columns=["source_file", "source_row", "id", "post_id", "issue", "detail"]
    )


def render_markdown_report(
    stats: dict[str, Any],
    parse_errors: list[tuple[str, str]],
    n_issues: int,
) -> str:
    """Sinh nội dung markdown báo cáo audit từ số liệu thực tế."""
    files = stats["input_files"]
    lines = [
        "# Báo cáo Audit dữ liệu",
        "",
        f"- Tổng số file annotation tìm thấy: {len(files)}",
        f"- Tổng bản ghi trước xử lý: {stats['total_records_before_processing']}",
        f"- Bản ghi hợp lệ: {stats['valid_records']}",
        f"- Bản ghi lỗi (loại có log): {stats['invalid_records']}",
        f"- Duplicate bị loại (có log): {stats['removed_duplicates']}",
        f"- Conflicting label (không sửa, chỉ báo cáo): {stats['conflicting_label_records']}",
        f"- Trùng văn bản khác id (chỉ báo cáo): {stats['duplicate_text_records']}",
        f"- Tổng dòng trong `data_issues.csv`: {n_issues}",
        "- Conservation check (hợp lệ + lỗi + trùng = tổng): "
        f"{'PASS' if stats['record_count_check'] else 'FAIL'}",
        "",
        "## Số bản ghi từng file",
        "",
        "| File | Số bản ghi |",
        "|---|---:|",
    ]
    for name, count in sorted(files.items()):
        lines.append(f"| {name} | {count} |")
    lines += [
        "",
        "## Phân bố nhãn (trên bản ghi hợp lệ)",
        "",
        "| Nhãn | Số lượng | Tỷ lệ |",
        "|---|---:|---:|",
        f"| 0 (non-toxic) | {stats['label_0']} | {stats['label_0_ratio_pct']}% |",
        f"| 1 (toxic) | {stats['label_1']} | {stats['label_1_ratio_pct']}% |",
        "",
        f"- Số lượng `post_id` duy nhất: {stats['unique_post_ids']}",
        f"- Số câu trung bình mỗi bài viết: {stats['avg_sentences_per_post']}",
    ]
    if parse_errors:
        lines += ["", "## Lỗi đọc file", ""]
        for name, message in parse_errors:
            lines.append(f"- **{name}**: {message}")
    lines += [
        "",
        "## Độ dài văn bản (trên bản ghi hợp lệ)",
        "",
        "| Chỉ số | Ký tự | Từ |",
        "|---|---:|---:|",
        f"| mean | {stats['text_char_length']['mean']} | {stats['text_word_length']['mean']} |",
        f"| median | {stats['text_char_length']['median']} | {stats['text_word_length']['median']} |",
        f"| min | {stats['text_char_length']['min']} | {stats['text_word_length']['min']} |",
        f"| max | {stats['text_char_length']['max']} | {stats['text_word_length']['max']} |",
        "",
        "## Chính sách xử lý",
        "",
        "- Bản ghi label trống/không thuộc {0,1}, text/post_id/id trống bị loại khỏi `merged.csv`, có truy vết trong `data_issues.csv`.",
        "- Trùng `id` hoặc trùng toàn bộ bản ghi: giữ bản ghi đầu tiên, log đầy đủ.",
        "- Cùng văn bản khác id hoặc khác nhãn: KHÔNG tự loại, KHÔNG sửa nhãn, chỉ báo cáo.",
        "- Nhiều câu cùng `post_id` là hành vi hợp lệ, không được coi là duplicate.",
        "- File annotation gốc không bị sửa.",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Pipeline audit
# ---------------------------------------------------------------------------
def run_audit(
    input_dir: Path, output_csv: Path, report_dir: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Thực hiện toàn bộ audit; trả về DataFrame gộp và stats.

    Raises:
        FileNotFoundError: nếu không có file CSV hợp lệ nào trong ``input_dir``.
        RuntimeError: nếu không còn bản ghi hợp lệ nào sau xử lý.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục đầu vào: {input_dir}")
    loaded = load_all_annotations(input_dir)
    frames = loaded["frames"]
    if not frames:
        if loaded["errors"]:
            detail = "; ".join(f"{n}: {m}" for n, m in loaded["errors"])
            raise FileNotFoundError(f"Không đọc được file annotation hợp lệ nào. Lỗi: {detail}")
        raise FileNotFoundError(f"Không có file CSV nào trong {input_dir}")

    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.sort_values(["source_file", "source_row"], kind="stable").reset_index(drop=True)

    tagged = tag_invalid_records(df_all)
    invalid = find_invalid(tagged)
    valid = tagged[tagged["issue"] == ""].drop(columns=["issue"]).copy()

    deduped, issue_log = find_duplicates(valid)
    if deduped.empty:
        raise RuntimeError(
            "Không còn bản ghi hợp lệ nào sau audit; không công bố merged.csv."
        )
    issue_rows = build_issue_rows(invalid, issue_log)

    # Label đã kiểm tra hợp lệ ở trên nên chuyển int an toàn, dtype thống nhất.
    deduped["label"] = deduped["_parsed_label"].astype(int)
    deduped = deduped.drop(columns=["_parsed_label"])

    extra_cols = [c for c in deduped.columns if c not in MERGED_COLUMN_ORDER]
    column_order = [c for c in MERGED_COLUMN_ORDER if c in deduped.columns] + extra_cols
    merged = deduped[column_order]

    per_file_counts = {str(f["source_file"].iloc[0]): len(f) for f in frames}
    stats = compute_statistics(
        per_file_counts=per_file_counts,
        total_records=len(df_all),
        invalid=invalid,
        valid=merged,
        issue_log=issue_log,
    )
    stats["record_count_check"] = (
        stats["valid_records"]
        + stats["invalid_records"]
        + stats["removed_duplicates"]
        == stats["total_records_before_processing"]
    )
    stats["issues_total"] = len(issue_rows)
    stats["parse_errors"] = [{"file": n, "error": m} for n, m in loaded["errors"]]

    report_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    # Ghi file tạm rồi atomic-swap để chạy lại không tạo trạng thái nửa vời.
    tmp_output = output_csv.with_name(output_csv.name + ".tmp")
    merged.to_csv(tmp_output, index=False, encoding="utf-8")
    tmp_output.replace(output_csv)

    (report_dir / "data_issues.csv").write_text(
        issue_rows.to_csv(index=False, encoding="utf-8-sig"), encoding="utf-8-sig"
    )
    (report_dir / "data_audit.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (report_dir / "data_audit.md").write_text(
        render_markdown_report(stats, loaded["errors"], len(issue_rows)), encoding="utf-8"
    )

    logger.info(
        "Audit xong: %d hợp lệ, %d lỗi, %d trùng đã loại; merged.csv = %s",
        stats["valid_records"], stats["invalid_records"],
        stats["removed_duplicates"], output_csv,
    )
    return merged, stats


def build_arg_parser() -> argparse.ArgumentParser:
    """Xây dựng parser CLI cho script audit."""
    parser = argparse.ArgumentParser(
        description="Audit và gộp dữ liệu annotation toxic/non-toxic."
    )
    parser.add_argument(
        "--input-dir", type=Path, default=REPO_ROOT / "annotations",
        help="Thư mục chứa các file annotation CSV.",
    )
    parser.add_argument(
        "--output-csv", type=Path,
        default=REPO_ROOT / "data" / "processed" / "merged.csv",
        help="Đường dẫn file gộp đầu ra.",
    )
    parser.add_argument(
        "--report-dir", type=Path, default=REPO_ROOT / "reports",
        help="Thư mục chứa báo cáo audit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    run_audit(args.input_dir, args.output_csv, args.report_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
