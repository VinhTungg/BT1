# Huấn luyện bằng resplit_v1

> Báo cáo này ghi kết quả bốn baseline của lượt trước. Mô hình hiện tại và
> kết quả tối ưu bước 2–3 nằm ở [steps23_v2/REPORT.md](steps23_v2/REPORT.md).

Run: `resplit_v1_20261101_081343`. Người dùng đã chọn dùng bộ chia resplit_v1. Mô hình hiện tại: **TF-IDF + Linear SVM**, chọn bằng Macro-F1 validation, không chọn bằng kết quả test.

## Bộ dữ liệu đang dùng

| Tập | Số câu | Số bài | Toxic | Tỷ lệ toxic |
|---|---:|---:|---:|---:|
| train | 6000 | 70 | 394 | 6.57% |
| val | 1277 | 16 | 83 | 6.50% |
| test | 1318 | 22 | 82 | 6.22% |

Đã giữ nguyên 8.595 ID sau lọc; không giao nhau về ID, ID số bài, text_clean hay text_tokenized giữa các tập. Ba bản ghi trùng non-toxic đã được liệt kê riêng khi tạo bản chia. Không sửa nhãn trong lượt huấn luyện này.

## Kết quả bốn baseline

| Mô hình | Macro-F1 val | Accuracy test | Precision toxic test | Recall toxic test | F1-toxic test | Macro-F1 test |
|---|---:|---:|---:|---:|---:|---:|
| bow_nb | 64.49% | 93.47% | 47.56% | 47.56% | 47.56% | 72.04% |
| bow_svm | 66.29% | 92.79% | 43.01% | 48.78% | 45.71% | 70.93% |
| tfidf_nb | 49.50% | 94.31% | 100.00% | 8.54% | 15.73% | 56.39% |
| tfidf_svm | 67.10% | 87.71% | 26.74% | 56.10% | 36.22% | 64.71% |
| dummy | 48.32% | 93.78% | 0.00% | 0.00% | 0.00% | 48.39% |

TF-IDF + SVM có validation Macro-F1 67,10%, F1-toxic 40,00%; cấu hình: `{"classifier__C": 0.1, "classifier__class_weight": "balanced", "vectorizer__min_df": 1, "vectorizer__ngram_range": [1, 1]}`. Trên test mới: TP=46, FN=36, FP=126, TN=1110.

BoW + NB có F1-toxic và Macro-F1 test cao nhất trong bốn baseline. Đây là quan sát sau đánh giá; lựa chọn theo validation vẫn được giữ. Không thay model theo kết quả test.

## Quy trình và kiểm tra

- Fit vectorizer và classifier từ đầu trên train 6.000 câu. Thử 72 cấu hình bốn baseline và một Dummy; 1 cấu hình BoW SVM không hội tụ được ghi lỗi và bỏ qua.
- Khóa toàn bộ cấu hình và tên mô hình thắng trong `final_evaluation/selection_before_test.json` trước khi dự đoán test. Dataset/model manifest và các artifact được đối chiếu SHA-256.
- 18 kiểm thử baseline/cải tiến/đánh giá và 2 kiểm thử chọn dataset mặc định đều đạt. Năm artifact được tải lại và thử đầu vào rỗng/từ lạ.

## Giới hạn của số đo

- Tập test đã đổi nên không so trực tiếp accuracy/F1 với bộ test 349 câu trước đó để kết luận mô hình tốt lên.
- Dữ liệu này từng xuất hiện trong lịch sử phát triển dự án. Đây là thực nghiệm mới với các nhóm bài tách biệt khi fit lại, không phải benchmark hoàn toàn mới so với mọi hoạt động trước đây.
- Toàn bộ 8.595 nhãn đã được nhóm chốt làm bộ nhãn chuẩn ngày 02/11/2026.
- Dummy luôn đoán non-toxic đạt accuracy 93,78% nhưng F1-toxic 0. Accuracy riêng lẻ không đủ đánh giá khả năng phát hiện toxic.

## Đường dẫn và chạy thử

- Bộ chia: `data/clean/{train,val,test}.csv`.
- Con trỏ dataset: `data/current_dataset.json`. Các CLI train mặc định đọc con trỏ này; tham số truyền trực tiếp có thể chỉ định thí nghiệm khác.
- Mô hình: `models/resplit_v1/baselines/`; con trỏ mô hình hiện tại: `models/current_run.json`.
- Các dữ liệu/artifact/báo cáo của lượt trước được giữ để đối chiếu. Snapshot con trỏ cũ: `archive/before_resplit_v1_activation_20261101_081343`.

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/predict_current.py "Cảm ơn bạn đã chia sẻ"
# Huấn luyện lại: nên chọn thư mục đầu ra mới để giữ artifact đã chốt
.\.venv\Scripts\python.exe -X utf8 scripts/train_baselines.py --model-dir models/resplit_v1/repeat --report-dir reports/resplit_v1/repeat
```

Trường hợp biên: chuỗi rỗng/từ chưa gặp phải dự đoán hợp lệ; đổi file dataset hoặc artifact làm sai hash phải bị từ chối.
