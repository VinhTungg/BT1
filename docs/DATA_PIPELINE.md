# Data Pipeline — Phân loại Toxic / Non-toxic tiếng Việt

Tài liệu mô tả toàn bộ pipeline dữ liệu từ file annotation của nhóm đến
`train.csv` / `val.csv` / `test.csv` dùng cho huấn luyện. Mọi con số trong
tài liệu là kết quả chạy thật trên dữ liệu trong `annotations/`.

## 1. Mục tiêu pipeline

Chuẩn hóa dữ liệu nhãn thủ công (2314 câu tiếng Việt từ 17 bài viết Facebook,
nhãn `0` = non-toxic, `1` = toxic) thành ba tập dữ liệu sạch, **không data
leakage theo bài viết**, có khả năng tái lập, hoàn chiếm một phần nguyên
vẹn của text gốc cho bước huấn luyện mô hình.

## 2. Dữ liệu đầu vào

- Thư mục `annotations/`, gồm 4 file do 4 thành viên gán nhãn thủ công
  (theo phân công trong `annotations/manifest.json`):
  `thanh_vien_1.csv` (579 dòng), `thanh_vien_2.csv` (579),
  `thanh_vien_3.csv` (578), `thanh_vien_4.csv` (578). **Tổng: 2314.**
- Encoding: UTF-8 có BOM (đọc bằng `utf-8-sig` nên có/không BOM đều được).
- Cột chung: `id` (mã băm nội dung, chuỗi hex, giữ nguyên số 0 đầu),
  `post_id` (định danh bài viết, dạng `2k5ptit:<id FB>`), `text`,
  `label`, `source_url`, `collected_at`, `extraction_method`, `note`.
- File thành viên 1 có thêm cột `status` (giá trị `approved`) và 2 cột
  dồn không tên do phân tách thừa (pipeline tự loại 2 cột rỗng không tên).
- `note` có nội dung ở file thành viên 2 (550 dòng) và 4 (67 dòng); được
  giữ lại nguyên vẹn trong mọi đầu ra.

## 3. Schema dữ liệu

`merged.csv`, `cleaned.csv` và các split dùng chung phần lõi:

```
id              chuỗi định danh bình luận (mã băm nội dung)
post_id         định danh bài viết — khóa nhóm để chống leakage
text            văn bản gốc (không bao giờ bị ghi đè)
text_clean      NFC + loại ký tự rác + chuẩn hóa khoảng trắng
text_tokenized  text_clean tách từ bằng underthesea
label           số nguyên 0 hoặc 1
source_file     file annotation gốc của bản ghi
source_row      số thứ tự dòng dữ liệu trong file gốc (1-based)
source_url      link bài viết (mất tham số theo dõi)
collected_at    thời điểm thu thập UTC
extraction_method  cách nhận diện của collector
note            ghi chú của người gán nhãn
status          (chỉ file thành viên 1)
```

## 4. Data auditing — `scripts/audit_data.py`

Chạy:

```bash
python scripts/audit_data.py
# tùy chọn: --input-dir, --output-csv, --report-dir
```

Mỗi file được đọc riêng (`dtype=str`, `keep_default_na=False` — không có
kají `NaN` nào biến thành chuỗi "nan"/"None"), sắp theo tên file nên thứ
tự listing OS không ảnh hưởng kết quả. Kiểm tra:

- Cột bắt buộc `id, post_id, text, label` tồn tại — thiếu thì file bị ghi
  lỗi vào báo cáo và không được dùng (các file hợp lệ còn lại vẫn xử lý).
- Label: chỉ "0"/"1" sau khi bỏ khoảng trắng (ví dụ `"0 "`, `" 1"` hợp lệ);
  trống hoặc giá trị khác (như "2") → bản ghi invalid, **không tự suy đoán
  thành 0/1**.
- `text`/`post_id`/`id` trống hoặc chỉ whitespace → bản ghi invalid.
- Trùng `id` trong file/giữa các file → giữ dòng đầu tiên, log từng bản
  ghi bị loại.
- Trùng toàn bộ bản ghi → tương tự.
- Trùng văn bản khác `id` / khác nhãn → **chỉ báo cáo**, không tự loại,
  không majority vote.
- Nhiều câu cùng `post_id` → hợp lệ, chỉ thống kê.

Kết quả chạy thật: 2314/2314 hợp lệ, 0 invalid, 0 duplicate, 0 conflicting;
label 2126 non-toxic (91.88%) / 188 toxic (8.12%); 17 post; trung bình
136.12 câu mỗi bài; độ dài text median 36 ký tự / 8 từ (min 1 → max 1507
ký tự). Conservation check: 2314 = 2314 + 0 + 0 → PASS.

Đầu ra: `data/processed/merged.csv`, `reports/data_audit.md`,
`reports/data_audit.json`, `reports/data_issues.csv`.

Pipeline không bao giờ ghi vào `annotations/` (file gốc chỉ đọc).

## 5. Quy tắc xử lý duplicate

- Cùng `id` (nội dung hash trùng) chỉ được xuất hiện một lần: lần xuất
  hiện đầu theo thứ tự (tên file, dòng) được giữ; phần còn lại bị loại
  với log đầy đủ (`duplicate_id`, `duplicate_full_record`).
- Văn bản trùng nhưng id khác nhau, hoặc văn bản trùng nhưng nhãn khác
  nhau: **giữ nguyên**, ghi vào `data_issues.csv` để nhóm review — không
  chọn nhãn theo đa số, không thử nghiệm thay thế nhãn.
- Số liệu thật: không phát hiện trường hợp nào (xem report audit).

## 6. Unicode normalization — `src/preprocessing.py`

`text_clean` bắt đầu từ `unicodedata.normalize("NFC", text)`: dấu phụ tổ
hợp (NFD) gộp về ký tự precomposed, bảo toàn toàn bộ dấu tiếng Việt
(á, ế, ệ, ả, ạ, đ…). NFC idempotent nên chạy lại không thay đổi text.

## 7. Text cleaning

Sau NFC, pipeline chỉ làm việc với "rác" không mang tín hiệu phân loại:

- Loại ký tự điều khiển (`Cc`), zero-width/invisible (`U+200B–U+200F`,
  `U+FEFF`, `U+2060`), chuyển khoảng trắng đặc biệt (NBSP, ideographic
  space…) về space thường, collapse khoảng trắng liên tiếp, trim.
- **Không xóa tín hiệu phân loại**: giữ nguyên dấu câu, emoji, URL,
  hashtag, mention (`[MENTION]`), từ phủ định ("không", "chẳng", "đừng"),
  từ toxic; không lower-case bắt buộc, không xoá stopwords, không
  stemming. Mọi tín hiệu phân loại được bảo toàn.

Text gốc `text` không bao giờ bị sửa; `text_clean` và `text_tokenized`
sống trong các cột riêng. `clean_text` idempotent (đã kiểm trên 2314 text).

## 8. Vietnamese word segmentation

`text_tokenized = word_tokenize(text_clean, format="text")` (underthesea),
xử lý từng văn bản độc lập:

```
Input:   "Sinh viên đang đóng học phí"
Output:  "Sinh_viên đang đóng học_phí"
```

Nếu underthesea lỗi với một văn bản, văn bản được giữ dạng `text_clean`
và warning được log (không mất bản ghi). `text_tokenized` chỉ dùng cho
phương pháp cần tách từ tiếng Việt; mô hình pretrained với tokenizer
tự nhiên nên dùng `text_clean` (hoặc `text`).

## 9. Group-aware splitting — `scripts/make_splits.py`

```bash
python scripts/make_splits.py
# tùy chọn: --input, --output-dir, --report-dir, --seed, --n-candidates
```

Seed mặc định `RANDOM_STATE = 42`. Tỷ lệ mục tiêu 70/15/15.

Bắt buộc: **tất cả câu cùng `post_id` nằm trong một tập duy nhất.**
Dataset có 17 post, trung bình 136 câu/bài (min 13, max 432) nên mức
sample 70/15/15 chính xác không thể đạt — group constraint được ưu tiên
tuyệt đối.

Thuật toán (tự thiết kế, có kiểm chứng): với mỗi candidate i (0..199),
tạo `random.Random(seed + i)` hoán vị các post_id, gán nhóm vào 3 bin
theo chiến lược *largest-remaining-bin* so với target số mẫu; score:
sai lệch tỷ lệ mẫu (x2) + sai lệch tỷ lệ toxic giữa tập (x5) + hình phạt
lớn nếu tập thiếu hẳn một lớp. Candidate score thấp nhất được chọn.
Mọi bước deterministic (tie-break: giữ candidate đầu).

Không dùng `train_test_split()` từng dòng. `StratifiedGroupKFold` không
được dùng vì với 17 nhóm nó không kiểm soát được tỷ lệ 70/15/15.

## 10. Cơ chế chống leakage

Sau chia, pipeline tự chạy và **fail cứng** nếu:

- `train/val/test` post sets có chồng lấn (3 phép `isdisjoint`).
- Multiset `id` giữa input và 3 tập không bảo toàn 1-1 (không mất/không
  sinh bản ghi).
- Tổng số bản ghi 3 tập = số bản ghi đầu vào.
- Còn `post_id` trống ở input → RuntimeError (không tự invent post_id); cần
  khôi phục từ dữ liệu nguồn trước.
- Xác nhận thêm trên dữ liệu thật: không trùng `id` giữa các tập,
  không exact duplicate text (`text_clean`, lower-case) giữa train/val,
  train/test, val/test.

Kết quả thật: mọi check PASS; không duplicate chéo; seed 42 cho
train 1594 / val 371 / test 349 = 2314 (68.88% / 16.03% / 15.08%), posts
9 / 4 / 4, toxic 125 / 32 / 31 (7.84% / 8.63% / 8.88%).

**Test set isolation**: test chỉ dùng cho đánh giá cuối. Preprocessing
trong pipeline (NFC, cleaning, tách từ underthesea) là hàm thuần túy
không học tham số,
nên áp dụng thống nhất cho cả test không gây leakage. Nếu mô hình cần
fit bất kỳ thành phần (TF-IDF, scaler…) thì chỉ fit trên train và
transform val/test.

## 11 / 12. Cách chạy từng phần & toàn pipeline

Từ repository root:

```bash
# Cài dependency cho pipeline (đã có pandas, scikit-learn trong môi trường)
python -m pip install underthesea pandas

# 1. Audit & merge dữ liệu thật
python scripts/audit_data.py

# 2. Tiền xử lý (NFC + cleaning + tách từ underthesea)
python -m src.preprocessing

# 3. Group-aware split + leakage verification + report
python scripts/make_splits.py
```

Kết quả idempotent: chạy lại toàn bộ pipeline sinh output byte-identical
(md5 `merged.csv`, `cleaned.csv`, ba split và `split_report.md` không đổi).

## 13. Cách chạy tests

```bash
python -m unittest discover -s tests -v
```

Hiện trạng: 69 test, 67 pass, 2 skip (2 test Playwright cũ của bộ thu
thập comments, không thuộc pipeline dữ liệu). Bao gồm:

- `tests/test_audit_data.py` — đọc hợp lệ, thiếu cột, thiếu label, label
  ngoài {0,1}, duplicate id (trong file/giữa file), duplicate text,
  conflicting label, thiếu post_id, file rỗng, lỗi encoding, không ghi
  đè file annotation gốc, conservation, giữ số 0 đầu.
- `tests/test_preprocessing.py` — NFC/NFD, whitespace, bảo toàn dấu,
  tách từ underthesea, dấu câu, emoji, URL, từ toxic, từ phủ định, text
  rỗng, input không phải chuỗi, idempotency, bảo toàn label/post_id.
- `tests/test_make_splits.py` — ràng buộc nhóm, tỷ lệ, distribution,
  group kích thước lệch, thiếu post_id, nhóm quá ít, dataset một lớp,
  input thiếu cột, deterministic, phát hiện trùng text giữa các tập.
- `tests/test_pipeline_integration.py` — chạy pipeline thật từ
  annotation copy vào thư mục tạm (audit → preprocess → split →
  reproducibility), không chạm dữ liệu gốc.

## 14. Giải thích output files

```
data/
├── processed/
│   ├── merged.csv     2314 dòng hợp lệ sau audit (schema nói ở mục 3)
│   └── cleaned.csv    merged + text_clean + text_tokenized
└── splits/            bàn giao cho huấn luyện
    ├── train.csv      1594 dòng — Người 2, Người 3 huấn luyện
    ├── val.csv         371 dòng — Người 2, Người 3 tinh chỉnh & chọn mô hình
    └── test.csv        349 dòng — Người 4 giữ, chỉ dùng đánh giá cuối

reports/
├── data_audit.md / data_audit.json   con số audit + chú thích
├── data_issues.csv                   log từng bản ghi lỗi/suspicious (rỗng = sạch)
├── split_report.md / split_report.json  chi tiết split + leakage checks
docs/
└── DATA_PIPELINE.md                  tài liệu này
```

Quy tắc bàn giao: **Không ai sử dụng test.csv để huấn luyện, chọn
hyperparameters, fit vocabulary/scaler hay điều chỉnh pipeline theo hiệu
năng test.** Test chỉ dùng để báo cáo kết quả cuối cùng.

## 15. Vấn đề còn tồn tại

- Số bài viết nguồn rất nhỏ (17) nên với seed 42 tỷ lệ 70/15/15 lệch
  tối đa ~0.6 điểm % trên val/test (15.43%/14.56%); thay seed chỉ đổi
  bằng `--seed`, ràng buộc nhóm luôn được giữ nguyên.
- Dataset mất cân bằng nặng (toxic 8.12%); model nên dùng class weighting
  tại bước huấn luyện — pipeline không oversample/undersample.
- Duplicate/near-duplicate chéo tập chỉ được kiểm exact + lower-case
  `text_clean`; nếu muốn fuzzy similarity (n-gram) thì thêm ở bước model
  với cảnh báo chỉ review, không tự sửa.
- Khoảng 550 dòng `note` (thành viên 2) và 67 (thành viên 4) được giữ
  nguyên; các note không ảnh hưởng label hoặc split.
