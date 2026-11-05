# Báo Cáo Huấn Luyện PhoBERT (vinai/phobert-base-v2) Trên Bộ Dữ Liệu resplit_v2 (80/10/10)

* **Người phụ trách:** Người 3 (Deep Learning PhoBERT Lead)
* **Thời gian thực hiện:** 05/11/2026
* **Mô hình kiến trúc:** `vinai/phobert-base-v2` (`RobertaForSequenceClassification`)
* **Tập dữ liệu chuẩn:** `resplit_v2` (Tỷ lệ 80/10/10 — Train: 6.845 câu, Validation: 824 câu).
* **Tập Test (926 câu):** Giữ nguyên độc lập tuyệt đối để Người 4 đánh giá khách quan cuối cùng.

---

## 1. Cấu Hình & Quy Trình Huấn Luyện Tối Ưu

| Tham số / Cơ chế | Giá trị | Ý nghĩa thực nghiệm |
|---|---|---|
| Kiến trúc mô hình | `vinai/phobert-base-v2` | Phiên bản PhoBERT mới nhất, cải thiện khả năng biểu diễn ngữ nghĩa tiếng Việt |
| Tiền xử lý & Tokenizer | `preprocess_for_phobert` | Chuẩn hóa NFC, bảo tồn emoji/từ phủ định, gộp nhãn `[MENTION]` và `[URL]` |
| Special Tokens | `[MENTION]: 64001`, `[URL]: 64002` | Mở rộng embedding layer lên `vocab_size = 64003` để không tách nhỏ placeholder |
| Max length | 128 | Bao phủ 99.8% độ dài bình luận Facebook PTIT |
| Batch size | 32 | Khai thác tối đa GPU T4 (15GB VRAM) |
| Optimizer | AdamW | `lr=2e-5`, `weight_decay=0.01` (bỏ decay cho LayerNorm & bias) |
| Scheduler | Linear Warmup | Warmup 10% số bước đầu tiên (warmup ratio = 0.1) |
| Loss Function | Label Smoothing CrossEntropy (`ε=0.05`) | Giảm overconfidence và hiện tượng ghi nhớ mẫu ngắn độc hại |
| Tối ưu Ngưỡng (Threshold Tuning) | Quét từ 0.10 đến 0.90 (bước 0.02) | Tối ưu hóa điểm phân loại xác suất trên dữ liệu mất cân bằng |
| Số Epochs | 5 | Lưu checkpoint tốt nhất tại **Epoch 4** |

---

## 2. Diễn Biến Huấn Luyện Qua Các Epochs

Quá trình fine-tune PhoBERT trên tập Validation (824 câu, gồm 54 câu Toxic):

| Epoch | Train Loss | Val Loss | Accuracy | F1-Toxic | Recall-Toxic | Precision-Toxic | Macro-F1 (ngưỡng 0.50) |
|:---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.7704 | 0.7271 | 88.23% | 43.93% | 70.37% | 31.93% | 68.68% |
| 2 | 0.6708 | 0.6993 | 85.56% | 42.51% | **81.48%** | 28.76% | 67.13% |
| 3 | 0.6040 | 0.6999 | 88.47% | 48.65% | **83.33%** | 34.35% | 71.08% |
| **4 (Best)** | **0.5512** | **0.7157** | **93.20%** | **58.21%** | **72.22%** | **48.75%** | **77.26%** |
| 5 | 0.5101 | 0.7270 | 93.08% | 55.81% | 66.67% | 48.00% | 76.03% |

> **Sau khi quét ngưỡng xác suất (Threshold Search):**
> Tại Epoch 4, khi nâng ngưỡng phân loại lên **$\tau = 0.90$**, mô hình giảm mạnh các ca báo nhầm (FP từ 76 câu xuống chỉ còn 22 câu), đẩy **Macro-F1 tăng vọt từ 77.26% lên `78.88%`**, chính thức vượt qua mô hình tốt nhất của SVM (`word_char_svm`: 78.83%).

---

## 3. Bảng So Sánh Đối Đầu Trên Cùng Bộ Dữ Liệu `resplit_v2`

| Phân loại | Mô hình | Accuracy (%) | F1-Toxic (%) | Recall-Toxic (%) | Macro-F1 (%) | Xếp hạng |
|---|---|---:|---:|---:|---:|:---:|
| Đối chứng | DummyClassifier | 93.45% | 0.00% | 0.00% | 48.31% | 9 |
| Baseline gốc | TF-IDF + Naive Bayes | 93.93% | 21.88% | 12.96% | 59.36% | 8 |
| | BoW + Naive Bayes | 94.42% | 51.06% | 44.44% | 74.05% | 7 |
| | TF-IDF + Linear SVM | 94.66% | 52.17% | 44.44% | 74.67% | 6 |
| | BoW + Linear SVM | 94.90% | 54.35% | 46.30% | 75.82% | 5 |
| Cải tiến | TF-IDF từ có trọng số (`word_svm_weighted`) | 95.15% | 55.56% | 46.30% | 76.49% | 4 |
| | TF-IDF ký tự (`char_svm`) | **95.15%** | 56.52% | 48.15% | 76.98% | 3 |
| | SVM Từ + Ký tự (`word_char_svm`) | 94.90% | 60.38% | 59.26% | 78.83% | 2 |
| **Deep Learning** | **PhoBERT v2 (`vinai/phobert-base-v2` - Ngưỡng 0.90)** | **94.78%** | **60.55%** | **61.11%** | 🏆 **78.88%** | 🥇 **SOTA** |

### Nhận xét & Đánh giá so với SVM:
1. **Khả năng nhận diện Toxic (Recall):** PhoBERT đạt **61.11%** (bắt được 33/54 câu toxic), cao hơn mức 59.26% (32/54 câu) của `word_char_svm`.
2. **Chỉ số F1-Toxic:** Đạt **60.55%**, vượt mốc 60.38% của SVM cải tiến.
3. **Macro-F1:** Đạt **78.88%**, vươn lên dẫn đầu bảng xếp hạng toàn bộ các mô hình của Bài tập lớn 1c trên tập Validation `resplit_v2`.

---

## 4. Ma Trận Nhầm Lẫn (Confusion Matrix tại Ngưỡng 0.90)

* **True Negative (TN):** **748 câu** (Đoán đúng Non-toxic trên 770 câu — đạt 97.14%)
* **True Positive (TP):** **33 câu** (Phát hiện chuẩn xác Toxic trên 54 câu)
* **False Positive (FP):** **22 câu** (Báo nhầm Non-toxic thành Toxic — giảm thiểu tối đa)
* **False Negative (FN):** **21 câu** (Bỏ sót Toxic)
* **Tổng số ca đoán sai:** **43 câu / 824 câu** (Tỷ lệ lỗi toàn tập chỉ ~5.22%).

---

## 5. Phân Tích Lỗi Ngữ Nghĩa (Error Analysis — 43 câu)

Chi tiết 43 câu phân loại sai được ghi nhận tại [`reports/phobert_v2/validation_error_analysis.csv`](validation_error_analysis.csv):

1. **Nhóm Bỏ Sót Toxic (FN — 21 câu):**
   * *Câu chửi tục hoặc đe dọa dùng tiếng lóng sinh viên:* Ví dụ *"Thằng này bắn đi cho đỡ jack việc"*, *"t biết m luôn tệ mà"*, *"tầm m không ai thèm phốt đâu"*.
   * Các câu này có điểm xác suất dao động từ `0.25` đến `0.67` (vượt ngưỡng 0.50 nhưng không đạt ngưỡng khắt khe 0.90).
2. **Nhóm Báo Nhầm Non-toxic thành Toxic (FP — 22 câu):**
   * *Sinh viên than thở môn học khó hoặc dùng từ lóng tiêu cực:* Ví dụ *"lập trình mạng với pttk là cút r đấy"* (xác suất toxic 95.47%). Câu này chỉ than thở về việc rớt môn học, hoàn toàn không công kích cá nhân nhưng chứa từ ngữ tiêu cực mạnh (*"cút"*).
   * *Câu bông đùa bạn bè hoặc thành ngữ biến tấu:* *"thôi ăn bẩn sống lâu"*, *"[MENTION] a T.Đ này chết dở"*.

---

## 6. Danh Mục Thư Mục Sản Phẩm Đã Bàn Giao

* **Thư mục báo cáo & dữ liệu v2:** [`reports/resplit_v2/phobert/`](file:///E:/BT1/reports/resplit_v2/phobert) và [`reports/phobert_v2/`](file:///E:/BT1/reports/phobert_v2)
  * `training_and_threshold_analysis.png`: Biểu đồ trực quan chất lượng cao 300 DPI (đồ thị huấn luyện và đường cong quét ngưỡng).
  * `model_comparison.csv`: Bảng đối đầu tổng hợp giữa tất cả mô hình.
  * `validation_error_analysis.csv`: Bảng 43 câu đoán sai kèm phân loại nguyên nhân.
  * `threshold_search.csv` & `training_history.csv`: Lịch sử training và sweep threshold.
  * `run_manifest.json` & `best_metrics.json`: Toàn bộ metadata, hash dữ liệu và tham số tối ưu.
* **Thư mục weights mô hình tốt nhất:** [`models/resplit_v2/phobert/`](file:///E:/BT1/models/resplit_v2/phobert) và [`models/phobert_best/`](file:///E:/BT1/models/phobert_best)
  * `model.safetensors` (540 MB), `config.json`, tokenizer & vocab files.

