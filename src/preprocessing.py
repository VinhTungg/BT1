# -*- coding: utf-8 -*-
"""Tiền xử lý văn bản tiếng Việt cho bài toán toxic classification.

Thiết kế bảo toàn ngữ nghĩa: chỉ chuẩn hóa Unicode NFC, làm sạch ký tự rác
(điều khiển, zero-width, khoảng trắng đặc biệt), chuẩn hóa khoảng trắng và
tách từ tiếng Việt bằng underthesea. KHÔNG xóa dấu câu, emoji, hashtag, URL,
từ toxic, từ phủ định, stopwords và KHÔNG stemming/lower-case bắt buộc — các
yếu tố này có thể chứa tín hiệu phân loại important.

Có thể tái sử dụng cho train/val/test và cho inference (hàm thuần túy,
không học tham số nào từ dữ liệu — nằm ngoài "fit" nên an toàn áp dụng cho
test set mà không gây leakage).

Chạy trực tiếp trên dữ liệu đã merge:

    python -m src.preprocessing --input data/processed/merged.csv \
        --output data/processed/cleaned.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd
from underthesea import word_tokenize

REPO_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("preprocessing")

# Ký tự zero-width/invisible thường sinh rác khi thu thập từ web.
_INVISIBLE_CHARS = (
    "\u200b",  # zero width space
    "\u200c",  # zero width non-joiner
    "\u200d",  # zero width joiner
    "\u200e",  # left-to-right mark
    "\u200f",  # right-to-left mark
    "\u2060",  # word joiner
    "\ufeff",  # zero width no-break space (BOM)
)
# Các ký tự khoảng trắng đặc biệt (white space Unicode) quy về space
# thường, kể cả NBSP, ideographic space, en/em space...
_SPECIAL_SPACES = frozenset(
    "\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006"
    "\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000"
)


def normalize_unicode_nfc(text: str) -> str:
    """Chuẩn hóa Unicode NFC, giữ nguyên dấu tiếng Việt.

    Ví dụ: tổ hợp "e" + "´" (NFD) về thành "é" đơn nguyên (NFC), còn "đ',
    "ờ", "ễ"... đều được bảo toàn.

    Raises:
        TypeError: nếu ``text`` không phải ``str``.
    """
    if not isinstance(text, str):
        raise TypeError(f"Văn bản đầu vào phải là str, nhận {type(text).__name__}")
    return unicodedata.normalize("NFC", text)


def remove_invisible_chars(text: str) -> str:
    """Loại ký tự điều khiển và zero-width không mang nội dung.

    Giữ ``\\t`` và ``\\n`` (được quy về space ở bước normalize_whitespace).
    Không loại dấu câu, emoji, ký tự tiếng Việt.
    """
    cleaned_chars: list[str] = []
    for ch in text:
        if ch in _INVISIBLE_CHARS:
            continue
        if ch in _SPECIAL_SPACES:
            cleaned_chars.append(" ")
            continue
        if unicodedata.category(ch) == "Cc":
            continue  # control char (bao gồm \r, không mang nội dung)
        cleaned_chars.append(ch)
    return "".join(cleaned_chars)


def normalize_whitespace(text: str) -> str:
    """Chuẩn hóa khoảng trắng: collapse liên tiếp về một space và trim.

    Tab/newline nhiều dòng được quy về một space; giữ khoảng trắng bên
    trong nội dung.
    """
    return " ".join(text.split())


def clean_text(text: str) -> str:
    """Chuẩn hóa NFC + loại ký tự rác + chuẩn hóa khoảng trắng.

    Hàm thuần túy và idempotent: ``clean_text(clean_text(x))`` luôn bằng
    ``clean_text(x)``. Đây là ``text_clean`` của schema.

    Raises:
        TypeError: nếu ``text`` không phải ``str``.
    """
    if not isinstance(text, str):
        raise TypeError(f"Văn bản đầu vào phải là str, nhận {type(text).__name__}")
    return normalize_whitespace(remove_invisible_chars(normalize_unicode_nfc(text)))


def segment_vietnamese(text: str) -> str:
    """Tách từ tiếng Việt bằng underthesea trên text đã clean.

    Dùng ``word_tokenize(text, format="text")`` — từ ghép được nối bằng
    dấu underscore. Xử lý từng văn bản độc lập. Nếu
    underthesea lỗi với một văn bản, log warning và trả về chính văn bản
    đầu vào (không mất bản ghi, không làm hỏng dữ liệu). Đây là
    ``text_tokenized`` của schema; không dùng mặc định cho mọi mô hình —
    một số tokenizer pretrained cần văn bản tự nhiên.

    Raises:
        TypeError: nếu ``text`` không phải ``str``.
    """
    if not isinstance(text, str):
        raise TypeError(f"Văn bản đầu vào phải là str, nhận {type(text).__name__}")
    normalized = clean_text(text)
    if not normalized:
        return ""
    try:
        return word_tokenize(normalized, format="text")
    except Exception:  # noqa: BLE001 - tokenizer có thể lỗi với input dị format
        logger.warning(
            "underthesea tokenize lỗi, giữ nguyên văn bản đã clean: %r",
            normalized[:60],
        )
        return normalized


def preprocess_text(text: str) -> tuple[str, str]:
    """Trả về ``(text_clean, text_tokenized)`` cho một văn bản."""
    cleaned = clean_text(text)
    return cleaned, segment_vietnamese(cleaned)


def preprocess_texts(texts: Iterable[str]) -> tuple[list[str], list[str]]:
    """Tiền xử lý mạnh nhiều văn bản giữ nguyên thứ tự."""
    cleaned_list: list[str] = []
    tokenized_list: list[str] = []
    for text in texts:
        cleaned, tokenized = preprocess_text(text)
        cleaned_list.append(cleaned)
        tokenized_list.append(tokenized)
    return cleaned_list, tokenized_list


# ---------------------------------------------------------------------------
# Áp dụng cho DataFrame
# ---------------------------------------------------------------------------
def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Thêm ``text_clean`` và ``text_tokenized`` vào DataFrame, bảo toàn schema.

    Kiểm tra sau xử lý:
      * Số dòng không đổi, không thêm mất bản ghi.
      * ``label`` và ``post_id`` không bị thay đổi.
      * Không phát sinh ``text_clean`` rỗng ở dòng có text gốc khác rỗng.
    """
    created_col = "_preprocess_tmp_clean"
    token_col = "_preprocess_tmp_token"
    try:
        cleaned, tokenized = preprocess_texts(df[text_col].astype("string").fillna(""))
    except Exception:
        df.drop(columns=[created_col, token_col], errors="ignore", inplace=True)
        raise

    working = df.copy()
    working[created_col] = cleaned
    working[token_col] = tokenized

    if len(working) != len(df):
        raise RuntimeError("preprocess_dataframe làm mất/thêm bản ghi.")
    if not _series_equal(working["label"], df["label"]):
        raise RuntimeError("preprocess_dataframe làm thay đổi cột label.")
    if not _series_equal(working["post_id"], df["post_id"]):
        raise RuntimeError("preprocess_dataframe làm thay đổi cột post_id.")

    mask_orig = working[text_col].astype("string").fillna("").str.strip() != ""
    empty_after = mask_orig & (working[created_col].astype("string").fillna("").str.strip() == "")
    if empty_after.any():
        n_bad = int(empty_after.sum())
        raise RuntimeError(
            f"preprocess_dataframe tạo ra {n_bad} text_clean rỗng ở bản ghi có text gốc."
        )

    working = working.drop(columns=["text_clean", "text_tokenized"], errors="ignore")
    working = working.rename(columns={created_col: "text_clean", token_col: "text_tokenized"})
    return working


def _series_equal(left: pd.Series, right: pd.Series) -> bool:
    """So sánh hai cột theo giá trị, coi ``NaN`` trống là bằng nhau."""
    return left.astype("string").fillna("").reset_index(drop=True).equals(
        right.astype("string").fillna("").reset_index(drop=True)
    )


def run_preprocessing(input_csv: Path, output_csv: Path) -> pd.DataFrame:
    """Đọc merged.csv, tiền xử lý toàn bộ, ghi cleaned.csv sau khi kiểm tra.

    Xác thực sau preprocess trước khi ghi (ghi file tạm rồi atomic-swap),
    để output cũ không bị hỏng nếu bước mới thất bại.
    """
    if not input_csv.exists():
        raise FileNotFoundError(f"Không tìm thấy file đầu vào: {input_csv}")
    frame = pd.read_csv(input_csv, encoding="utf-8", dtype=str, keep_default_na=False)
    required = {"id", "post_id", "text", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"File đầu vào thiếu cột bắt buộc: {sorted(missing)}")

    processed = preprocess_dataframe(frame)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output_csv.with_name(output_csv.name + ".tmp")
    processed.to_csv(tmp_output, index=False, encoding="utf-8")
    tmp_output.replace(output_csv)

    # Kiểm tra idempotency trên mẫu thật: chạy lại clean một lần nữa không đổi.
    sample = processed["text_clean"].head(200)
    re_cleaned = [clean_text(t) for t in sample]
    if re_cleaned != list(sample):
        raise RuntimeError("clean_text không idempotent trên dữ liệu thật.")
    logger.info("Preprocessing xong: %d dòng -> %s", len(processed), output_csv)
    return processed


def build_arg_parser() -> argparse.ArgumentParser:
    """Xây dựng parser CLI cho chạy trực tiếp module."""
    parser = argparse.ArgumentParser(
        description="Tiền xử lý văn bản tiếng Việt (NFC, cleaning, tách từ underthesea)."
    )
    parser.add_argument(
        "--input", type=Path, default=REPO_ROOT / "data" / "processed" / "merged.csv",
        help="File CSV đầu vào (merged.csv).",
    )
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "data" / "processed" / "cleaned.csv",
        help="File CSV đầu ra (cleaned.csv).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    run_preprocessing(args.input, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
