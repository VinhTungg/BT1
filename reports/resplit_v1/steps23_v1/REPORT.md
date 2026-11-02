# Tối ưu bước 2 và 3 trên validation

Nhãn đã được nhóm chốt. Giữ nguyên ID, nhãn và phân chia dữ liệu.
Train 6000, validation 1277. Test không được đọc trong lượt này.
Lưới cố định: 104 cấu hình, 0 cấu hình lỗi.
Chọn Macro-F1; hòa điểm xét F1-toxic, Recall-toxic, rồi cấu hình đầu tiên.

| Mô hình | Accuracy | Precision toxic | Recall toxic | F1-toxic | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| bow_nb | 0.9319 | 0.4565 | 0.2530 | 0.3256 | 0.6449 |
| bow_svm | 0.9264 | 0.4154 | 0.3253 | 0.3649 | 0.6629 |
| tfidf_nb | 0.9350 | 0.5000 | 0.0120 | 0.0235 | 0.4950 |
| tfidf_svm | 0.8943 | 0.3169 | 0.5422 | 0.4000 | 0.6710 |
| dummy | 0.9350 | 0.0000 | 0.0000 | 0.0000 | 0.4832 |
| preserve_bow_nb | 0.9327 | 0.4737 | 0.3253 | 0.3857 | 0.6750 |
| preserve_bow_svm | 0.9123 | 0.3670 | 0.4819 | 0.4167 | 0.6846 |
| preserve_tfidf_nb | 0.9374 | 0.8000 | 0.0482 | 0.0909 | 0.5292 |
| preserve_tfidf_svm | 0.9193 | 0.3913 | 0.4337 | 0.4114 | 0.6841 |
| preserve_char_svm | 0.9139 | 0.3579 | 0.4096 | 0.3820 | 0.6679 |
| preserve_word_char_svm | 0.9154 | 0.3737 | 0.4458 | 0.4066 | 0.6805 |
| slang_bow_nb | 0.9342 | 0.4912 | 0.3373 | 0.4000 | 0.6826 |
| slang_bow_svm | 0.9107 | 0.3578 | 0.4699 | 0.4062 | 0.6790 |
| slang_tfidf_nb | 0.9374 | 0.8000 | 0.0482 | 0.0909 | 0.5292 |
| slang_tfidf_svm | 0.8982 | 0.3333 | 0.5663 | 0.4196 | 0.6819 |
| slang_char_svm | 0.8958 | 0.3106 | 0.4940 | 0.3814 | 0.6623 |
| slang_word_char_svm | 0.9162 | 0.3800 | 0.4578 | 0.4153 | 0.6851 |

Tốt nhất: **slang_word_char_svm**; chênh Macro-F1 so với baseline: **+1.41 điểm phần trăm**.
Đặc trưng: `{"kind": "word_char", "ngram_range": [1, 2], "min_df": 2, "char_weight": 1.0}`.
Classifier: `{"C": 0.1, "class_weight": "balanced"}`.

`preserve`: chuẩn hóa an toàn, giữ phủ định/dấu câu/emoji, gộp placeholder thành một token.
`slang`: thêm từ điển cố định cho ko/k0/khum/hok/khong/đc/thik/cx/rùi/nma; không thay k, r, đk hay 100k.
BoW và TF-IDF thử cùng ngram/min_df. TF-IDF dùng sublinear TF, IDF làm mịn và L2.
Nhánh ký tự dùng char_wb; nhánh kết hợp dùng từ + ký tự với trọng số cố định trong lưới.
`controlled_comparison.csv` giữ unigram, min_df=1, alpha=0.1 hoặc C=0.1/balanced để tách ảnh hưởng đặc trưng và tiền xử lý.
Từ vựng và IDF chỉ fit train. Artifact nhận văn bản thô và chứa toàn bộ tiền xử lý.
Số câu validation trùng biểu diễn sau quy tắc chuẩn hóa: `{"preserve": {"text_clean": 0, "text_tokenized": 0}, "slang": {"text_clean": 1, "text_tokenized": 1}}`. Đây là va chạm do ánh xạ cố định; không gộp câu, không sửa nhãn và không học từ validation.
Điểm validation đã được dùng để chọn cấu hình; chưa phải kết quả đánh giá cuối trên test.
Không dùng điểm test trong các lượt cũ để chọn cấu hình của lượt này.
