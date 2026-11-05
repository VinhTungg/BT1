# Thí nghiệm cải tiến ML baseline

- Chế độ: full; seed 42; 60 cấu hình mới.
- Train 6845 câu; validation 824 câu. Không đánh giá test.
- Chọn Macro-F1, rồi F1-toxic, Recall-toxic; hòa hoàn toàn giữ cấu hình đầu tiên.
- Ngưỡng quyết định SVM giữ ở 0. Không thu thập thêm hoặc sửa nhãn.

| Mô hình | Nhánh | Accuracy | Precision toxic | Recall toxic | F1 toxic | Macro-F1 | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bow_nb | original | 0.9442 | 0.6000 | 0.4444 | 0.5106 | 0.7405 | 16 | 30 |
| bow_svm | original | 0.9490 | 0.6579 | 0.4630 | 0.5435 | 0.7582 | 13 | 29 |
| tfidf_nb | original | 0.9393 | 0.7000 | 0.1296 | 0.2188 | 0.5936 | 3 | 47 |
| tfidf_svm | original | 0.9466 | 0.6316 | 0.4444 | 0.5217 | 0.7467 | 14 | 30 |
| dummy | original | 0.9345 | 0.0000 | 0.0000 | 0.0000 | 0.4831 | 0 | 54 |
| word_svm_weighted | improved | 0.9515 | 0.6944 | 0.4630 | 0.5556 | 0.7649 | 11 | 29 |
| char_svm | improved | 0.9515 | 0.6842 | 0.4815 | 0.5652 | 0.7698 | 12 | 28 |
| word_char_svm | improved | 0.9490 | 0.6154 | 0.5926 | 0.6038 | 0.7883 | 20 | 22 |

Tốt nhất trên validation: **word_char_svm**.
So với bow_svm: Macro-F1 +0.0300, F1-toxic +0.0603, FP +7, FN -7.

## Kiểm tra bổ sung trên ba fold theo bài viết trong train

Các cấu hình đã được chọn bằng validation; đây là chẩn đoán sau lựa chọn, không phải nested CV hay kết quả test.

| Mô hình | Fold thành công | Macro-F1 trung bình | Độ lệch chuẩn | F1-toxic trung bình |
|---|---:|---:|---:|---:|
| bow_svm | 3/3 | 0.6168 | 0.0227 | 0.2679 |
| word_svm_weighted | 3/3 | 0.6321 | 0.0237 | 0.2974 |
| char_svm | 3/3 | 0.6310 | 0.0277 | 0.2945 |
| word_char_svm | 3/3 | 0.6600 | 0.0201 | 0.3551 |

Chênh lệch Macro-F1 trung bình ba fold của ứng viên tốt nhất so với baseline gốc: +0.0432.

Giữ bốn baseline gốc cho bảng thực nghiệm bắt buộc, bàn giao ứng viên mới như mô hình bổ sung.

## Cấu hình tốt nhất từng nhánh

- `word_svm_weighted`: `{"classifier__C": 0.1, "classifier__class_weight": {"0": 1, "1": 4}, "features__word__ngram_range": [1, 1]}`
- `char_svm`: `{"classifier__C": 0.1, "classifier__class_weight": {"0": 1, "1": 4}}`
- `word_char_svm`: `{"classifier__C": 0.1, "classifier__class_weight": {"0": 1, "1": 8}}`

Validation nhỏ và đã được dùng nhiều lần để chọn cấu hình; chưa thể khẳng định mức tăng sẽ giữ nguyên trên dữ liệu mới.
Pipeline mới nhận DataFrame chứa `text_clean` và `text_tokenized`, khác đầu vào danh sách chuỗi của baseline gốc.
Bàn giao cả baseline gốc và thí nghiệm mới cho Người 4; dùng test độc lập khi chốt đánh giá.
