# Kết quả ML baseline trên validation

- Chế độ: full; seed: 42.
- Train: 6000 câu; validation: 1277 câu.
- Đầu vào: `text_tokenized`; nhãn 0 = non-toxic, 1 = toxic.
- Chọn Macro-F1 cao nhất; hòa điểm xét F1-toxic, Recall-toxic, rồi cấu hình đầu tiên.
- Chỉ fit trên train; chưa đánh giá test và không refit trên train+val.

| Mô hình | Accuracy | Precision toxic | Recall toxic | F1 toxic | Macro-F1 | TN / FP / FN / TP |
|---|---:|---:|---:|---:|---:|---|
| bow_nb | 0.9319 | 0.4565 | 0.2530 | 0.3256 | 0.6449 | 1169 / 25 / 62 / 21 |
| bow_svm | 0.9264 | 0.4154 | 0.3253 | 0.3649 | 0.6629 | 1156 / 38 / 56 / 27 |
| tfidf_nb | 0.9350 | 0.5000 | 0.0120 | 0.0235 | 0.4950 | 1193 / 1 / 82 / 1 |
| tfidf_svm | 0.8943 | 0.3169 | 0.5422 | 0.4000 | 0.6710 | 1097 / 97 / 38 / 45 |
| dummy | 0.9350 | 0.0000 | 0.0000 | 0.0000 | 0.4832 | 1194 / 0 / 83 / 0 |

## Tham số được chọn

- `bow_nb`: `{"classifier__alpha": 0.1, "vectorizer__min_df": 1, "vectorizer__ngram_range": [1, 1]}`
- `bow_svm`: `{"classifier__C": 1.0, "classifier__class_weight": "balanced", "vectorizer__min_df": 1, "vectorizer__ngram_range": [1, 1]}`
- `tfidf_nb`: `{"classifier__alpha": 0.1, "vectorizer__min_df": 1, "vectorizer__ngram_range": [1, 2]}`
- `tfidf_svm`: `{"classifier__C": 0.1, "classifier__class_weight": "balanced", "vectorizer__min_df": 1, "vectorizer__ngram_range": [1, 1]}`
- `dummy`: `{}`

Các điểm trên được dùng để chọn mô hình, không phải kết quả đánh giá độc lập cuối cùng.
Xem `validation_trials.csv` cho toàn bộ cấu hình và `val_predictions_*.csv` để phân tích lỗi.
Baseline tốt nhất theo tiêu chí đã chốt: `tfidf_svm`.
Manifest ghi hash đầu vào, phiên bản thư viện và kiểm tra lưu/tải lại toàn bộ 1277 dự đoán validation.
