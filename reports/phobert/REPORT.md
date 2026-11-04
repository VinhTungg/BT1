# Báo Cáo Huấn Luyện PhoBERT (vinai/phobert-base-v2) — Người 3

* **Thời gian hoàn thành:** 04/11/2026
* **Mô hình:** `vinai/phobert-base-v2`
* **Người phụ trách:** Người 3 (Deep Learning PhoBERT Lead)
* **Tập dữ liệu:** `resplit_v1` (Train: 6.000 câu, Validation: 1.277 câu). Dữ liệu được chia theo `post_id` chống rò rỉ dữ liệu.
* **Tập Test (1.318 câu):** Giữ nguyên độc lập bàn giao cho Người 4 đánh giá cuối cùng.

---

## 1. Thiết lập Huấn luyện

| Tham số | Giá trị | Giải thích |
|---|---|---|
| Cột đầu vào | `text_tokenized` | Đã được Người 1 tách từ chuẩn bằng `underthesea` (từ ghép nối `_`) |
| Max length | 128 | Bao phủ >95% bình luận Facebook sinh viên PTIT |
| Batch size | 32 | Phù hợp với GPU T4 (Colab) |
| Optimizer | AdamW | `lr=2e-5`, `weight_decay=0.01` (bỏ decay cho LayerNorm & bias) |
| Scheduler | Linear Warmup | Warmup ratio = 0.1 (10% số bước đầu tiên) |
| Loss Function | Weighted CrossEntropyLoss | Trọng số lớp: Non-toxic (0) = 0.5351, Toxic (1) = 7.6142 |
| Số Epochs | 5 | Tránh overfitting trên tập văn bản ngắn |

---

## 2. Kết quả trên tập Validation (1.277 câu)

Mô hình đạt điểm **Macro-F1 cao nhất tại Epoch 2**:

| Chỉ số | Giá trị | Nhận xét |
|---|---:|---|
| **Macro-F1** | **67.21%** | **Vượt mốc Baseline tốt nhất của Người 2 (TF-IDF + SVM: 67.10%)** |
| **F1-Toxic** | **40.18%** | Tương đương SVM (40.00%) |
| **Recall-Toxic** | **54.22%** | **Bắt được 45/83 câu toxic** (rất cao so với BoW NB 25.3% và BoW SVM 32.5%) |
| **Precision-Toxic** | 31.91% | Chấp nhận đánh đổi một số FP để tối đa hóa khả năng phát hiện toxic |
| **Accuracy** | 89.51% | Độ chính xác tổng thể |
| **Validation Loss** | 0.5203 | Hội tụ tốt |

### Ma trận nhầm lẫn (Confusion Matrix):
* **True Negative (TN):** 1.098 câu (Đoán đúng Non-toxic)
* **True Positive (TP):** 45 câu (Đoán đúng Toxic)
* **False Positive (FP):** 96 câu (Báo nhầm Toxic)
* **False Negative (FN):** 38 câu (Bỏ sót Toxic)

---

## 3. Bảng So Sánh Đối Đầu với Các Mô Hình của Người 2

| Mô hình | Accuracy (%) | F1-Toxic (%) | Recall-Toxic (%) | Macro-F1 (%) |
|---|---:|---:|---:|---:|
| DummyClassifier (Mốc đối chứng) | 93.50% | 0.00% | 0.00% | 48.32% |
| BoW + Naive Bayes (Người 2) | 93.19% | 32.56% | 25.30% | 64.49% |
| BoW + Linear SVM (Người 2) | 92.64% | 36.49% | 32.53% | 66.29% |
| TF-IDF + Naive Bayes (Người 2) | 93.50% | 2.35% | 1.20% | 49.50% |
| TF-IDF + Linear SVM (Người 2 - Best Baseline) | 89.43% | 40.00% | 54.22% | 67.10% |
| **PhoBERT (`vinai/phobert-base-v2` - Người 3)** | **89.51%** | **40.18%** | **54.22%** | **67.21%** |
| *Tối ưu Từ+Ký tự + OOF (Người 2 - steps23_v2)* | 91.62% | 42.33% | 48.19% | 68.86% |

---

## 4. Phân Tích Lỗi (Error Analysis)

Chi tiết 134 câu dự đoán sai được ghi nhận tại `reports/phobert/validation_error_analysis.csv`:
* **False Positives (96 câu):** Thường rơi vào các bình luận phàn nàn về học phí, thi cử hoặc hệ thống tín chỉ có chứa các từ ngữ cảm thán gay gắt (ví dụ: *"chán vl"*, *"phòng đào tạo làm ăn như hạch"*). Mặc dù sinh viên bức xúc nhưng theo quy ước nhãn không mang tính xúc phạm, lăng mạ cá nhân.
* **False Negatives (38 câu):** Rơi vào các câu toxic ngắn, sử dụng teencode/slang viết tắt chưa có trong ngữ cảnh PhoBERT (ví dụ: *"đm"*, *"óc c*" viết tắt), hoặc các câu mỉa mai ngầm cần ngữ cảnh bài viết gốc.

---

## 5. Sản Phẩm Bàn Giao cho Người 4

1. **Checkpoint mô hình:** [models/phobert_best/](file:///E:/BT1/models/phobert_best) (`model.safetensors`, `config.json`, `tokenizer_config.json`, `vocab.txt`, `bpe.codes`).
2. **Biểu đồ huấn luyện:** [reports/phobert/training_curves_phobert.png](file:///E:/BT1/reports/phobert/training_curves_phobert.png).
3. **Ma trận nhầm lẫn:** [reports/phobert/confusion_matrix_phobert.png](file:///E:/BT1/reports/phobert/confusion_matrix_phobert.png).
4. **Bảng phân tích lỗi:** [reports/phobert/validation_error_analysis.csv](file:///E:/BT1/reports/phobert/validation_error_analysis.csv).
5. **Log số đo JSON:** [reports/phobert/best_metrics.json](file:///E:/BT1/reports/phobert/best_metrics.json).

