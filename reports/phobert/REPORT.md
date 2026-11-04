# Báo Cáo Huấn Luyện PhoBERT (vinai/phobert-base-v2) — Người 3

* **Thời gian hoàn thành:** 04/11/2026
* **Mô hình kiến trúc:** `vinai/phobert-base-v2`
* **Người phụ trách:** Người 3 (Deep Learning PhoBERT Lead)
* **Tập dữ liệu:** `resplit_v1` (Train: 6.000 câu, Validation: 1.277 câu). Dữ liệu được chia theo `post_id` để chống rò rỉ dữ liệu giữa các bài viết.
* **Tập Test (1.318 câu):** Giữ nguyên độc lập bàn giao cho Người 4 đánh giá độc lập cuối cùng.

---

## 1. Thiết lập Huấn luyện

| Tham số | Giá trị | Giải thích |
|---|---|---|
| Cột đầu vào | `text_tokenized` / `text` | Tiền xử lý chuẩn hóa thống nhất qua `preprocess_for_phobert` |
| Max length | 128 | Thiết kế nhằm bao phủ phần lớn bình luận Facebook PTIT trên Colab T4 |
| Batch size | 32 | Phù hợp với bộ nhớ GPU T4 (15GB VRAM) |
| Optimizer | AdamW | `lr=2e-5`, `weight_decay=0.01` (bỏ decay cho LayerNorm & bias) |
| Scheduler | Linear Warmup | Warmup ratio = 0.1 (10% số bước đầu tiên) |
| Loss Function | Weighted CrossEntropyLoss | Trọng số lớp: Non-toxic (0) = 0.5351, Toxic (1) = 7.6142 |
| Số Epochs | 5 | Kèm theo dõi Macro-F1 để lưu lại checkpoint tốt nhất |

> [!NOTE]
> Các kỹ thuật tối ưu như Mixed Precision (FP16) hay Dynamic Padding là mục tiêu thiết kế nhằm giảm tải VRAM và rút ngắn thời gian tính toán trên môi trường GPU Colab T4.

---

## 2. Kết quả trên tập Validation (1.277 câu)

Mô hình đạt điểm **Macro-F1 cao nhất tại Epoch 2**:

| Chỉ số | Giá trị | Nhận xét |
|---|---:|---|
| **Macro-F1** | **67.21%** | Nhỉnh hơn TF-IDF + SVM ban đầu (+0.11%), thấp hơn SVM tối ưu (-1.65%) |
| **F1-Toxic** | **40.18%** | Tương đương TF-IDF + SVM ban đầu (40.00%) |
| **Recall-Toxic** | **54.22%** | **Bắt được 45/83 câu toxic** (cao hơn đáng kể so với BoW NB 25.3% và SVM tối ưu 48.19%) |
| **Precision-Toxic** | 31.91% | Thấp hơn SVM tối ưu (37.74%) do phát hiện nhạy hơn |
| **Accuracy** | 89.51% | Thấp hơn SVM tối ưu (91.46%) |
| **Validation Loss** | 0.5203 | Đạt tối ưu tại Epoch 2 trước khi xuất hiện overfitting |

### Diễn biến quá trình huấn luyện & Hiện tượng Overfitting:
* **Epoch 1 $\rightarrow$ Epoch 2:** Train loss giảm từ 0.5912 xuống 0.4354; Val loss đạt 0.5203; Macro-F1 tăng từ 64.72% lên **67.21%** (đạt đỉnh).
* **Epoch 3 $\rightarrow$ Epoch 5:** Train loss tiếp tục giảm mạnh xuống 0.1887, nhưng Val loss tăng vọt lên 1.0621 và Macro-F1 chững lại quanh mức 65.7% – 66.7%.
* **Kết luận:** Mô hình có dấu hiệu quá khớp (overfitting) rõ rệt sau Epoch 2 trên tập dữ liệu bình luận ngắn. Do đó, việc khôi phục checkpoint tại **Epoch 2** (theo cơ chế early stopping / validation selection) là bắt buộc để đảm bảo khả năng tổng quát hóa tốt nhất.

### Ma trận nhầm lẫn (Confusion Matrix):
* **True Negative (TN):** 1.098 câu (Đoán đúng Non-toxic)
* **True Positive (TP):** 45 câu (Đoán đúng Toxic)
* **False Positive (FP):** 96 câu (Báo nhầm Non-toxic thành Toxic)
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
| *Tối ưu Từ+Ký tự + OOF (Người 2 - steps23_v2)* | **91.46%** | **42.33%** | **48.19%** | **68.86%** |

### Phân tích khách quan & Sự đánh đổi (Trade-off):
1. **So với TF-IDF + Linear SVM ban đầu:** PhoBERT có kết quả xấp xỉ (Macro-F1 67.21% vs 67.10%, nhỉnh hơn 0.11 điểm phần trăm).
2. **So với Mô hình SVM tối ưu (`steps23_v2`):** 
   - PhoBERT thấp hơn khoảng **1.65 điểm phần trăm Macro-F1** (67.21% so với 68.86%) và thấp hơn về Accuracy (89.51% so với 91.46%).
   - **Về khả năng phát hiện Toxic:** PhoBERT phát hiện thêm **5 câu toxic** (TP: 45 vs 40, Recall toxic: 54.22% vs 48.19%).
   - **Về độ chính xác (Precision):** Để bắt thêm 5 câu toxic, PhoBERT báo nhầm thêm **30 câu non-toxic** thành toxic (FP: 96 vs 66, Precision: 31.91% vs 37.74%).
3. **Đánh giá tổng thể:** Chưa có cơ sở để kết luận PhoBERT vượt trội toàn diện so với SVM tối ưu hiện tại. Thay vào đó, hai mô hình thể hiện sự đánh đổi rõ rệt giữa ưu tiên độ nhạy phát hiện toxic (PhoBERT) và độ chính xác tổng thể cân bằng (SVM `steps23_v2`).

---

## 4. Phân Tích Lỗi (Error Analysis)

Chi tiết 134 câu dự đoán sai được lưu trữ đầy đủ tại `reports/phobert/validation_error_analysis.csv`:
* **False Positives (96 câu):** Thường rơi vào các bình luận phàn nàn gay gắt về dịch vụ nhà trường (học phí, cổng đăng ký tín chỉ, bãi xe) có từ ngữ bức xúc (ví dụ: *"chán vl"*, *"làm ăn như hạch"*). Mặc dù có giọng điệu tiêu cực nhưng theo quy ước nhãn không có yếu tố xúc phạm danh dự cá nhân.
* **False Negatives (38 câu):** Rơi vào các bình luận chửi thề ngắn, tiếng lóng viết tắt hoặc câu mang hàm ý mỉa mai ngầm cần ngữ cảnh bài viết nguồn (ví dụ: viết tắt *"đm"*, châm chọc cá nhân không nhắc tên trực tiếp).

---

## 5. Danh Mục Sản Phẩm Bàn Giao cho Người 4

Tất cả sản phẩm đã được chuẩn hóa và lưu trữ đồng bộ:

1. **Cấu hình & Tokenizer Checkpoint:** [models/phobert_best/](../../models/phobert_best/) (`config.json`, `tokenizer_config.json`, `vocab.txt`, `bpe.codes`, `added_tokens.json`). File weights nhị phân lớn (`model.safetensors`) được lưu trữ riêng và tải qua Drive/Colab theo tài liệu hướng dẫn.
2. **Bảng dự đoán toàn bộ Validation:** [reports/phobert/val_predictions.csv](val_predictions.csv) (Đủ 1.277 câu, gồm `id`, `label`, `prediction`, `toxic_probability`, `outcome`).
3. **Bảng phân tích lỗi chi tiết:** [reports/phobert/validation_error_analysis.csv](validation_error_analysis.csv) (134 câu dự đoán sai).
4. **Lịch sử huấn luyện từng Epoch:** [reports/phobert/training_history.csv](training_history.csv) (Train loss, val loss, macro-F1, F1 toxic qua 5 epoch).
5. **Kết quả quét ngưỡng quyết định:** [reports/phobert/threshold_search.csv](threshold_search.csv) (Quét ngưỡng từ 0.10 đến 0.90).
6. **Thông tin định danh lần chạy (Run Manifest):** [reports/phobert/run_manifest.json](run_manifest.json) (Kiến trúc, seed, hash dữ liệu, siêu tham số).
7. **Biểu đồ trực quan hóa:**
   * [reports/phobert/training_curves_phobert.png](training_curves_phobert.png) (Đường cong Loss và F1 qua các epoch).
   * [reports/phobert/confusion_matrix_phobert.png](confusion_matrix_phobert.png) (Ma trận nhầm lẫn tại checkpoint tốt nhất).
8. **Bảng so sánh baseline:** [reports/phobert/model_comparison.csv](model_comparison.csv).
