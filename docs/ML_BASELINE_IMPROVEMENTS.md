# Nhánh thử đặc trưng ký tự và trọng số lớp

`train_improved_baselines.py` là công cụ thử nghiệm bổ sung trên dataset
đang chọn. Nó giữ nguyên bốn baseline và không thay mô hình hiện tại.
Lượt tối ưu đã chốt hiện nằm trong
[steps23_v2/REPORT.md](../reports/resplit_v1/steps23_v2/REPORT.md).

## Các nhánh

| Nhánh | Đặc trưng |
|---|---|
| `word_svm_weighted` | TF-IDF từ, thử unigram hoặc unigram+bigram |
| `char_svm` | TF-IDF char_wb 3–5, sublinear TF |
| `word_char_svm` | Ghép TF-IDF từ và ký tự |

Script thử C và trọng số lớp, chọn theo validation, rồi chạy chẩn đoán ba
fold theo bài viết trên train. Đây là chẩn đoán sau chọn cấu hình, không
phải ước lượng nested CV độc lập. Không đọc test.

## Chạy lượt mới

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/train_improved_baselines.py --model-dir models/resplit_v1/improvements_repeat --report-dir reports/resplit_v1/improvements_repeat
```

Nguồn tham chiếu mặc định là `models/resplit_v1/baselines/manifest.json`;
SHA-256 train/validation phải khớp. Script nạp lại mô hình tham chiếu để xác
nhận số đo và giữ nguyên các artifact baseline.

Các tùy chọn gồm `--train`, `--val`, `--baseline-manifest`, `--model-dir`,
`--report-dir`, `--seed`, `--quick`. Nếu không chỉ định, dữ liệu và thư mục
đầu ra được giải quyết qua `data/current_dataset.json`.

Đầu vào artifact của nhánh này là DataFrame chứa `text_clean` và
`text_tokenized`. Mô hình tối ưu hiện tại nhận văn bản thô qua
`scripts/predict_current.py`; xem [quy trình hiện tại](TEXT_FEATURE_OPTIMIZATION.md).

Các artifact thuộc bộ chia cũ đã được dọn khỏi thư mục làm việc. Nếu cần
đối chiếu lịch sử, xem [bản lưu cục bộ](../archive/README.md).
