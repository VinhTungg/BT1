# Báo Cáo Huấn Luyện 5-Fold GroupKFold Cho Các Mô Hình SVM

* **Ngày thực hiện:** 05/11/2026
* **Chiến lược:** 5-Fold StratifiedGroupKFold theo `post_id` (100% Group-Aware, không rò rỉ dữ liệu).
* **Tổng số mẫu:** 7.669 câu (95 bài viết, 499 câu toxic chiếm 6.51%).
* **Tập Test (926 câu):** Giữ nguyên độc lập 100% cho Người 4.

---

## 1. Bảng So Sánh Hiệu Năng 5-Fold Giữa Các Mô Hình SVM

| Mô hình | Macro-F1 Mean ± Std (%) | Macro-F1 OOF (%) | F1-Toxic OOF (%) | Recall Toxic OOF (%) | Precision Toxic OOF (%) | Accuracy OOF (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| BoW + Linear SVM | 64.7% ± 2.99% | 64.96% | 33.14% | 23.45% | 56.52% | 93.85% |
| TF-IDF + Linear SVM | 66.3% ± 1.86% | 66.44% | 36.12% | 26.85% | 55.14% | 93.82% |
| **Từ + Ký tự (`word_char_svm`)** | **68.94% ± 1.42%** | **68.98%** | **41.34%** | **34.67%** | **51.18%** | **93.6%** |

---

## 2. Chi Tiết Ma Trận Nhầm Lẫn OOF Của `word_char_svm` (Toàn Bộ 7.669 Câu)

* **True Negative (TN):** 7005 câu (Đoán chuẩn Non-toxic)
* **True Positive (TP):** 173 câu (Bắt được 173/499 câu Toxic)
* **False Positive (FP):** 165 câu (Báo nhầm)
* **False Negative (FN):** 326 câu (Bỏ sót)

---

## 3. Nhận Xét Quan Trọng Cho Người 3 (Để So Sánh Với PhoBERT)

1. Mô hình SVM tốt nhất (`word_char_svm`) đạt **Mean Macro-F1 = 68.94% ± 1.42%** và **OOF Macro-F1 = 68.98%**.
2. Độ lệch chuẩn giữa 5 Folds là **± 1.42%**, cho thấy sự ổn định rất cao khi chia K-Fold theo bài viết.
3. Khi bạn chạy xong 5 Folds PhoBERT trên Colab, hãy đối chiếu:
   - Nếu PhoBERT đạt **Mean Macro-F1 > 68.94%** hoặc **OOF Macro-F1 > 68.98%**, PhoBERT tiếp tục khẳng định vị thế dẫn đầu!
   - Đặc biệt, với kỹ thuật **Ensemble 5 Folds**, PhoBERT hứa hẹn sẽ bứt phá mạnh hơn nữa ở các câu toxic mỉa mai và ngữ cảnh phức tạp.
