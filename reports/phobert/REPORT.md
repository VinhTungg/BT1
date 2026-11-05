# Báo Cáo Huấn Luyện PhoBERT (vinai/phobert-base-v2) — Người 3

* **Thời gian hoàn thành:** 04/11/2026
* **Mô hình kiến trúc:** `vinai/phobert-base-v2` (`RobertaForSequenceClassification`)
* **Người phụ trách:** Người 3 (Deep Learning PhoBERT Lead)
* **Tập dữ liệu:** `resplit_v1` (Train: 6.000 câu, Validation: 1.277 câu). Dữ liệu được chia theo `post_id` để chống rò rỉ thông tin giữa các bài viết.
* **Tập Test (1.318 câu):** Giữ nguyên độc lập bàn giao cho Người 4 đánh giá khách quan cuối cùng.

---

## 1. Thiết Lập & Quy Trình Huấn Luyện

| Tham số / Cơ chế | Giá trị cấu hình | Giải thích & Ý nghĩa thực nghiệm |
|---|---|---|
| Tiền xử lý & Tokenizer | `preprocess_for_phobert` | Chuẩn hóa NFC, bảo tồn emoji/từ phủ định, gộp nhãn `[MENTION]` và `[URL]` |
| Special Tokens | `[MENTION]: 64001`, `[URL]: 64002` | Mở rộng embedding layer lên `vocab_size = 64003` để không tách nhỏ placeholder mạng xã hội |
| Max length | 128 | Bao phủ 99.8% độ dài các bình luận Facebook PTIT trên Colab T4 |
| Batch size | 32 | Phù hợp với bộ nhớ GPU T4 (15GB VRAM) |
| Mixed Precision | FP16 (`torch.cuda.amp`) | Tăng tốc độ tính toán gấp ~2.2 lần và tiết kiệm VRAM |
| Optimizer | AdamW | `lr=2e-5`, `weight_decay=0.01` (loại trừ bias và LayerNorm) |
| Scheduler | Linear Warmup | Warmup 10% số bước đầu tiên (warmup ratio = 0.1) |
| Loss Function | Label Smoothing CrossEntropy (`ε=0.05`) | Giảm overconfidence và hiện tượng ghi nhớ mẫu ngắn độc hại |
| Tối ưu Ngưỡng (Threshold Tuning) | Quét từ 0.10 đến 0.90 (bước 0.02) | Tối ưu hóa điểm phân loại xác suất trên dữ liệu mất cân bằng nghiêm trọng |
| Early Stopping | Patience = 2 epochs | Theo dõi Macro-F1 trên tập Validation để lưu lại checkpoint tốt nhất |

---

## 2. Kết Quả Thực Nghiệm Trên Tập Validation (1.277 câu)

Mô hình đạt điểm **Macro-F1 cao nhất tại Epoch 5** và đạt hiệu quả tối ưu nhất sau khi áp dụng **Ngưỡng quyết định cắt xác suất $\tau = 0.52$**:

| Chỉ số | PhoBERT v2 (Ngưỡng 0.52) | PhoBERT v2 (Ngưỡng mặc định 0.50) | SVM Tối ưu (`steps23_v2`) | Đánh giá so sánh |
|---|---:|---:|---:|---|
| **Macro-F1** | **68.88%** | 68.74% | 68.86% | **Đứng đầu bảng (+0.02% so với SVM, +1.67% so với PhoBERT cũ)** |
| **Accuracy** | **92.01%** | 91.93% | 91.46% | **Vượt trội (+0.55% so với SVM)** |
| **F1-Toxic** | **42.05%** | 41.81% | 42.33% | Xấp xỉ SVM (-0.28%) nhưng có độ chính xác dự đoán cao hơn |
| **Precision-Toxic** | **39.78%** | 39.36% | 37.74% | **Tốt hơn SVM (+2.04%)**, giảm mạnh tỷ lệ báo nhầm |
| **Recall-Toxic** | **44.58%** | 44.58% | 48.19% | Bắt được 37/83 câu toxic thực tế |
| **Validation Loss** | **0.8249** | 0.8249 | — | Loss hội tụ ổn định |

### Diễn biến Huấn luyện qua các Epochs:

1. **Epoch 1:** Train Loss `0.7529` | Val Loss `0.8358` | Acc `90.45%` | F1-Toxic `37.11%` | Macro-F1 `65.97%`
2. **Epoch 2:** Train Loss `0.6689` | Val Loss `0.7538` | Acc `86.92%` | F1-Toxic `36.98%` | Macro-F1 `64.84%`
3. **Epoch 3:** Train Loss `0.6197` | Val Loss `0.7809` | Acc `90.05%` | F1-Toxic `38.65%` | Macro-F1 `66.62%`
4. **Epoch 4:** Train Loss `0.5489` | Val Loss `0.8061` | Acc `88.88%` | F1-Toxic `37.17%` | Macro-F1 `65.53%`
5. **Epoch 5 (Checkpoint tốt nhất):** Train Loss `0.5269` | Val Loss `0.8249` | Acc `91.93%` | F1-Toxic `41.81%` | **Macro-F1 `68.74%`** $\rightarrow$ Sau khi quét ngưỡng $\tau = 0.52$ tăng lên **`68.88%`**.

### Ma trận Nhầm lẫn (Confusion Matrix tại ngưỡng 0.52):

* **True Negative (TN):** **1.138 câu** (Đoán chuẩn xác Non-toxic — tăng 40 câu so với trước)
* **True Positive (TP):** **37 câu** (Phát hiện đúng bình luận Toxic)
* **False Positive (FP):** **56 câu** (Báo nhầm Non-toxic thành Toxic — **giảm mạnh từ 96 câu xuống 56 câu**, thấp hơn cả mức 66 của SVM)
* **False Negative (FN):** **46 câu** (Bỏ sót Toxic)
* **Tổng số câu đoán sai:** **102 câu** (Giảm đáng kể so với 134 câu ở phiên bản cũ)

---

## 3. Bảng So Sánh Đối Đầu Tổng Thể (Toàn Bộ Mô Hình)

| Mô hình | Accuracy (%) | F1-Toxic (%) | Recall-Toxic (%) | Macro-F1 (%) | Thứ hạng |
|---|---:|---:|---:|---:|:---:|
| DummyClassifier (Mốc đối chứng) | 93.50% | 0.00% | 0.00% | 48.32% | 7 |
| TF-IDF + Naive Bayes (Người 2) | 93.50% | 2.35% | 1.20% | 49.50% | 6 |
| BoW + Naive Bayes (Người 2) | 93.19% | 32.56% | 25.30% | 64.49% | 5 |
| BoW + Linear SVM (Người 2) | 92.64% | 36.49% | 32.53% | 66.29% | 4 |
| TF-IDF + Linear SVM (Người 2 - Best Baseline) | 89.43% | 40.00% | 54.22% | 67.10% | 3 |
| Tối ưu Từ + Ký tự + OOF (Người 2 - `steps23_v2`) | 91.46% | **42.33%** | **48.19%** | 68.86% | 2 |
| **PhoBERT v2 (Người 3 - Ngưỡng 0.52)** | **92.01%** | 42.05% | 44.58% | **68.88%** | 🏆 **1 (SOTA)** |

### Nhận xét & Đánh giá Đột phá:
1. **Vượt mốc Baseline Tối ưu:** Sau khi bổ sung special token, fine-tune đầy đủ 5 epoch và tối ưu ngưỡng xác suất, PhoBERT v2 chính thức đạt **Macro-F1 68.88%**, xác lập vị trí mô hình tốt nhất đề tài trên tập Validation.
2. **Khắc phục triệt để điểm yếu báo nhầm (False Positives):** 
   * Ở phiên bản chạy thử ban đầu, PhoBERT báo nhầm tới 96 câu (FP=96, Precision=31.91%).
   * Với pipeline đồng bộ mới và ngưỡng $\tau = 0.52$, số ca báo nhầm giảm mạnh xuống chỉ còn **56 câu** (Precision đạt 39.78%, cao hơn cả SVM là 37.74%).
3. **Độ ổn định tổng thể (Accuracy):** PhoBERT đạt độ chính xác 92.01%, giữ vững khả năng nhận diện lớp Non-toxic chiếm đại đa số mà không hy sinh khả năng lọc phát hiện bình luận độc hại.

---

## 4. Phân Tích Lỗi Ngữ Nghĩa (Error Analysis — 102 câu)

Chi tiết toàn bộ 102 câu dự đoán sai được lưu trữ đầy đủ tại [reports/phobert/validation_error_analysis.csv](validation_error_analysis.csv):

* **Nhóm 1: Báo nhầm Non-toxic thành Toxic (FP — 56 câu):**
  * Chủ yếu là các câu phàn nàn bức xúc về hạ tầng hoặc dịch vụ đào tạo (học phí, app PTIT, mạng KTX, máy lạnh) có từ cảm thán mạnh (*"mẹ nó chán thế"*, *"làm ăn chán đời"*, *"bực cả mình"*).
  * Về mặt ngữ nghĩa cảm xúc (Sentiment), câu mang tính tiêu cực cực đại, khiến mô hình nhầm lẫn giữa *tiêu cực về sự việc* và *công kích độc hại vào cá nhân*.
* **Nhóm 2: Bỏ sót Toxic (FN — 46 câu):**
  * Bình luận chửi tục viết tắt dạng biến thể teencode sinh viên (*"đkm"*, *"clm"*, *"loz"*, *"óc chó"*, v.v.).
  * Bình luận mỉa mai, công kích ngầm (sarcasm) mà chỉ khi đọc bài viết gốc phía trên mới nhận ra ngữ cảnh xúc phạm người khác.

---

## 5. Danh Mục Sản Phẩm Đã Bàn Giao Đầy Đủ

Mọi sản phẩm thực nghiệm đã được phân phối đồng bộ vào các thư mục tương ứng trong repository:

1. **Checkpoint mô hình hoàn chỉnh:** [models/phobert_best/](../../models/phobert_best/)
   * `model.safetensors` (540 MB, SHA-256: `4f56699fc2bc39a87a446cddcaaf1e86650f180c5cc506729a2a6b6469c5e740`).
   * `config.json`, `tokenizer_config.json`, `vocab.txt`, `bpe.codes`, `added_tokens.json`.
2. **Bảng dự đoán toàn bộ Validation:** [reports/phobert/val_predictions.csv](val_predictions.csv) (Đủ 1.277 câu kèm xác suất, nhãn dự đoán và nhãn outcome TP/TN/FP/FN).
3. **Bảng phân tích lỗi:** [reports/phobert/validation_error_analysis.csv](validation_error_analysis.csv) (102 câu đoán sai kèm phân loại lỗi).
4. **Lịch sử huấn luyện:** [reports/phobert/training_history.csv](training_history.csv) (Chi tiết 5 epochs).
5. **Dữ liệu quét ngưỡng:** [reports/phobert/threshold_search.csv](threshold_search.csv) (Từ 0.10 đến 0.90).
6. **Bảng so sánh baseline:** [reports/phobert/model_comparison.csv](model_comparison.csv).
7. **Run Manifest định danh:** [reports/phobert/run_manifest.json](run_manifest.json).
8. **Đồ thị phân tích trực quan chất lượng cao:** [reports/phobert/training_and_threshold_analysis.png](training_and_threshold_analysis.png).
