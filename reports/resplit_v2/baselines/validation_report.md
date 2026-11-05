# Kết quả ML baseline trên validation

- Chế độ: full; seed: 42.
- Train: 6845 câu; validation: 824 câu.
- Đầu vào: `text_tokenized`; nhãn 0 = non-toxic, 1 = toxic.
- Chọn Macro-F1 cao nhất; hòa điểm xét F1-toxic, Recall-toxic, rồi cấu hình đầu tiên.
- Chỉ fit trên train; chưa đánh giá test và không refit trên train+val.

| Mô hình | Accuracy | Precision toxic | Recall toxic | F1 toxic | Macro-F1 | TN / FP / FN / TP |
|---|---:|---:|---:|---:|---:|---|
| bow_nb | 0.9442 | 0.6000 | 0.4444 | 0.5106 | 0.7405 | 754 / 16 / 30 / 24 |
| bow_svm | 0.9490 | 0.6579 | 0.4630 | 0.5435 | 0.7582 | 757 / 13 / 29 / 25 |
| tfidf_nb | 0.9393 | 0.7000 | 0.1296 | 0.2188 | 0.5936 | 767 / 3 / 47 / 7 |
| tfidf_svm | 0.9466 | 0.6316 | 0.4444 | 0.5217 | 0.7467 | 756 / 14 / 30 / 24 |
| dummy | 0.9345 | 0.0000 | 0.0000 | 0.0000 | 0.4831 | 770 / 0 / 54 / 0 |

## Tham số được chọn

- `bow_nb`: `{"classifier__alpha": 0.1, "vectorizer__min_df": 1, "vectorizer__ngram_range": [1, 1]}`
- `bow_svm`: `{"classifier__C": 0.1, "classifier__class_weight": "balanced", "vectorizer__min_df": 1, "vectorizer__ngram_range": [1, 2]}`
- `tfidf_nb`: `{"classifier__alpha": 0.1, "vectorizer__min_df": 2, "vectorizer__ngram_range": [1, 1]}`
- `tfidf_svm`: `{"classifier__C": 1.0, "classifier__class_weight": "balanced", "vectorizer__min_df": 1, "vectorizer__ngram_range": [1, 2]}`
- `dummy`: `{}`

Các điểm trên được dùng để chọn mô hình, không phải kết quả đánh giá độc lập cuối cùng.
Xem `validation_trials.csv` cho toàn bộ cấu hình và `val_predictions_*.csv` để phân tích lỗi.
Baseline tốt nhất theo tiêu chí đã chốt: `bow_svm`.
Manifest ghi hash đầu vào, phiên bản thư viện và kiểm tra lưu/tải lại toàn bộ 824 dự đoán validation.
