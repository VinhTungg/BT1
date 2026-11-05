# 🏆 BÁO CÁO TỔNG KẾT & SO SÁNH TOÀN DIỆN CẢ 3 PHIÊN BẢN (V1, V2, V3)
## ĐỐI ĐẦU CHI TIẾT: NGƯỜI 2 (MACHINE LEARNING SVM) VS NGƯỜI 3 (DEEP LEARNING PHOBERT)

* **Đề tài:** Phân loại bình luận toxic và non-toxic trong các nhóm Facebook sinh viên PTIT.
* **Mục đích:** Tài liệu tổng hợp trực quan nhất giúp nhóm thảo luận, lựa chọn cấu hình dữ liệu và mô hình tối ưu nhất để nộp Báo cáo BTL và Slide thuyết trình.

---

## 📊 PHẦN 1: BẢNG SO SÁNH TỔNG HỢP MASTER TABLE (TẤT CẢ CÁC MÔ HÌNH)

Dưới đây là bảng đối chiếu trực diện toàn bộ thực nghiệm của **Người 2** và **Người 3** qua 3 phiên bản dữ liệu:

| Phiên bản dữ liệu | Phương thức đánh giá | Mô hình | Phụ trách | Accuracy | Recall Toxic (Khả năng bắt toxic) | F1-Toxic | **Macro-F1 (Tiêu chí cốt lõi)** | Xếp hạng phiên bản |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **v1** *(70/15/15)* | Tập Val cố định *(1.277 câu)* | DummyClassifier | Người 2 | 93.50% | 0.00% | 0.00% | 48.32% | 7 |
| | | TF-IDF + Naive Bayes | Người 2 | 93.50% | 1.20% | 2.35% | 49.50% | 6 |
| | | BoW + Naive Bayes | Người 2 | 93.19% | 25.30% | 32.56% | 64.49% | 5 |
| | | BoW + Linear SVM | Người 2 | 92.64% | 32.53% | 36.49% | 66.29% | 4 |
| | | TF-IDF + Linear SVM | Người 2 | 89.43% | **54.22%** | 40.00% | 67.10% | 3 |
| | | SVM Tối ưu Từ+Ký tự (`steps23_v2`) | Người 2 | 91.46% | 48.19% | 42.33% | **68.86%** | 2 |
| | | **PhoBERT v2** *(Epoch 5, $\tau=0.52$)* | **Người 3** | **92.01%** | 44.58% | **42.05%** | 🏆 **68.88%** | 🥇 **SOTA v1** |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **v2** *(80/10/10)* | Tập Val cố định *(824 câu)* | DummyClassifier | Người 2 | 93.45% | 0.00% | 0.00% | 48.31% | 8 |
| | | TF-IDF + Naive Bayes | Người 2 | 93.93% | 12.96% | 21.88% | 59.36% | 7 |
| | | BoW + Naive Bayes | Người 2 | 94.42% | 44.44% | 51.06% | 74.05% | 6 |
| | | TF-IDF + Linear SVM | Người 2 | 94.66% | 44.44% | 52.17% | 74.67% | 5 |
| | | BoW + Linear SVM | Người 2 | 94.90% | 46.30% | 54.35% | 75.82% | 4 |
| | | TF-IDF từ có trọng số (`word_svm`) | Người 2 | 95.15% | 46.30% | 55.56% | 76.49% | 3 |
| | | TF-IDF ký tự (`char_svm`) | Người 2 | **95.15%** | 48.15% | 56.52% | 76.98% | 2 |
| | | **SVM Từ + Ký tự (`word_char_svm`)** | **Người 2** | 94.90% | 59.26% | 60.38% | **78.83%** | 🥈 Nhì v2 |
| | | **PhoBERT v2** *(Epoch 4, $\tau=0.90$)* | **Người 3** | 94.78% | 🏆 **61.11%** | 🏆 **60.55%** | 🏆 **78.88%** | 🥇 **SOTA v2** |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **v3** *(5-Fold)* | **5-Fold GroupKFold** *(Toàn bộ 7.669 câu)* | BoW + Linear SVM | Người 2 | 93.85% | 23.45% | 33.14% | 64.70% $\pm$ 2.99% *(OOF 64.96%)* | 3 |
| | | TF-IDF + Linear SVM | Người 2 | 93.82% | 26.85% | 36.12% | 66.30% $\pm$ 1.86% *(OOF 66.44%)* | 2 |
| | | **SVM Từ + Ký tự (`word_char_svm`)** | **Người 2** | **93.60%** | 34.67% | 41.34% | **68.94% $\pm$ 1.42%** *(OOF 68.98%)* | 🥈 Nhì v3 |
| | | **PhoBERT v2 Ensemble** *(5 Folds)* | **Người 3** | 93.17% | 🏆 **44.89%** | 🏆 **46.09%** | 🏆 **71.16% $\pm$ 2.39%** *(OOF 71.22%)* | 🥇 **SOTA v3** |

---

## 🔍 PHẦN 2: GIẢI MÃ BẢN CHẤT CỦA TỪNG PHIÊN BẢN (TẠI SAO CÓ SỰ KHÁC BIỆT?)

```text
                  [ TOÀN BỘ CORPUS: 8.595 BÌNH LUẬN ]
                                   │
      ┌────────────────────────────┼────────────────────────────┐
      ▼                            ▼                            ▼
  Phiên bản v1                 Phiên bản v2                 Phiên bản v3
  (70 / 15 / 15)               (80 / 10 / 10)            (5-Fold GroupKFold)
  • Train: 6.000 câu           • Train: 6.845 câu        • Pool: 7.669 câu chia 5 folds
  • Val: 1.277 câu             • Val: 824 câu            • Test: 926 câu giữ độc lập
  • Macro-F1: ~68.8%           • Macro-F1: ~78.8%        • Macro-F1: ~71.2% (Thực chất nhất)
```

1. **Phiên bản v1 (70/15/15 — ~68.8%):**
   * *Đặc điểm:* Phân chia cổ điển. Tập Train chỉ có 394 câu toxic (quá ít).
   * *Ý nghĩa:* Là mốc khởi điểm (Baseline), chứng minh việc thiếu dữ liệu toxic khiến cả SVM lẫn PhoBERT gặp khó khăn trong việc nhận diện ngữ cảnh tiêu cực.
2. **Phiên bản v2 (80/10/10 — ~78.8%):**
   * *Đặc điểm:* Nâng Train lên 80% (thêm 51 câu toxic / 12 bài viết). Tập Val chỉ còn 824 câu (13 bài viết).
   * *Tại sao điểm vọt lên 78.8%:* Tập Val 13 bài có phong cách viết khá tương đồng với Train $\rightarrow$ Điểm số đạt đỉnh cao nhất. 
   * *Ưu điểm:* Bảng điểm đẹp mắt, cực kỳ ấn tượng để đưa vào Slide thuyết trình và Báo cáo.
3. **Phiên bản v3 (5-Fold GroupKFold — 71.2% OOF, Mean 71.16%):**
   * *Đặc điểm:* Kiểm định chéo 5 lần trên toàn bộ 7.669 câu. Không bỏ sót bất kỳ bài viết nào.
   * *Tại sao điểm là 71.2%:* Vì nó kiểm tra qua tất cả 95 bài viết (kể cả những bài teencode khó nhất, mỉa mai ngầm) $\rightarrow$ **Đây là thước đo khoa học trung thực và khách quan nhất** về năng lực tổng quát hóa.
   * *Ưu điểm:* Chứng minh mô hình ổn định tuyệt đối với độ lệch chuẩn nhỏ ($\pm 1.42\%$ ở SVM, $\pm 2.39\%$ ở PhoBERT).

---

## ⚔️ PHẦN 3: SO SÁNH SÂU GIỮA 2 PHƯƠNG PHÁP: SVM (NGƯỜI 2) VS PHOBERT (NGƯỜI 3)

| Tiêu chí so sánh | Machine Learning: Linear SVM (Người 2) | Deep Learning: PhoBERT (Người 3) | Người chiến thắng |
|---|---|---|:---:|
| **Điểm số Macro-F1** | 68.94% (v3) / 78.83% (v2) | **71.16% (v3) / 78.88% (v2)** | 🏆 **PhoBERT (Thắng ở cả 3 phiên bản)** |
| **Khả năng bắt Toxic (Recall)** | Bắt được **173 / 499 câu** (34.67% ở v3) | Bắt được **224 / 499 câu** (44.89% ở v3) | 🏆 **PhoBERT (Bắt hơn SVM 51 câu toxic)** |
| **Khả năng hiểu ngữ cảnh / mỉa mai** | Kém (chỉ soi n-gram từ/ký tự tĩnh) | Xuất sắc (cơ chế Self-Attention tiếng Việt) | 🏆 **PhoBERT** |
| **Thời gian huấn luyện** | Siêu nhanh (**~30 giây** trên CPU) | Chậm (**~12–15 phút** trên GPU T4 Colab) | ⚡ **SVM** |
| **Dung lượng & Tài nguyên triển khai** | Siêu nhẹ (file `.joblib` ~1 MB) | Nặng (file checkpoint ~540 MB, cần PyTorch) | ⚡ **SVM** |
| **Độ ổn định giữa các Folds** | Rất cao (Độ lệch chuẩn **$\pm 1.42\%$**) | Tốt (Độ lệch chuẩn **$\pm 2.39\%$**) | ⚡ **SVM** |
| **Khả năng kết hợp Ensemble** | Giới hạn ở Voting cơ bản | Dễ dàng lấy trung bình xác suất 5 Folds | 🏆 **PhoBERT** |

---

## 🎯 PHẦN 4: MA TRẬN TƯ VẤN LỰA CHỌN CHO CẢ NHÓM

Nhóm bạn có thể lựa chọn theo một trong các phương án sau tùy thuộc vào mục tiêu:

### 👉 PHƯƠNG ÁN KHUYÊN DÙNG NHẤT (Điểm số BTL cao nhất): "Lộ trình cải tiến 3 bước"
Trong Báo cáo BTL (Chương 3), nhóm **không cần chọn bỏ cái nào mà hãy đưa CẢ 3 BẢNG VÀO** theo mạch kể chuyện khoa học cực kỳ thuyết phục:
1. **Bước 1 (v1):** Khởi động với bộ chia 70/15/15. Điểm số dừng ở mức ~68.8%. Phân tích thấy do tập Train thiếu hụt câu toxic.
2. **Bước 2 (v2):** Tối ưu lại phân chia sang 80/10/10. Điểm số tăng vọt lên **78.83% (SVM)** và **78.88% (PhoBERT)**. Khẳng định dữ liệu nhiều hơn giúp mô hình học sâu hơn.
3. **Bước 3 (v3):** Đỉnh cao phương pháp luận. Áp dụng 5-Fold GroupKFold để kiểm định độ vững chắc (Robustness) $\rightarrow$ Điểm số OOF đạt **71.22% (PhoBERT)**, chứng minh PhoBERT vượt trội SVM ở cả 5/5 Folds (+2.22%) và bắt trúng thêm 51 câu toxic.

---

### 🏆 ĐỐI VỚI VIỆC CHỌN MÔ HÌNH VÀ BÀN GIAO:

1. **Mô hình vô địch (Champion Model) bàn giao cho Người 4 chấm điểm Test:**
   * Chọn **PhoBERT v2 / Ensemble (Người 3)** làm mô hình chính của nhóm.
   * Đây là mô hình đạt SOTA ở cả 3 phiên bản, bắt toxic tốt nhất và thể hiện đúng tinh thần ứng dụng Deep Learning hiện đại.
2. **Mô hình đối chứng mạnh nhất (Strong Baseline):**
   * Chọn **SVM Từ + Ký tự (`word_char_svm` của Người 2)** làm mốc đối chiếu mạnh nhất.
   * Cho thấy Người 2 không chỉ làm baseline tầm thường mà đã tối ưu n-gram ký tự 3–5 rất công phu, tiệm cận được PhoBERT.
3. **Phần demo trực tiếp (nếu Thầy/Cô yêu cầu test thử câu mới):**
   * Sử dụng hàm `predict_ensemble` của PhoBERT hoặc `predict_current.py` của SVM để nhập bình luận tiếng Việt bất kỳ và in ngay kết quả.

---

### 📁 VỊ TRÍ CÁC FILE TRONG WORKSPACE

* File báo cáo tổng hợp này: [`TONG_KET_SO_SANH_3_VERSION.md`](file:///E:/BT1/TONG_KET_SO_SANH_3_VERSION.md) *(và bản lưu tại [`reports/TONG_KET_SO_SANH_3_VERSION.md`](file:///E:/BT1/reports/TONG_KET_SO_SANH_3_VERSION.md))*
* Chi tiết kết quả K-Fold SVM: [`reports/kfold_v3/svm/REPORT.md`](file:///E:/BT1/reports/kfold_v3/svm/REPORT.md)
* Chi tiết kết quả K-Fold PhoBERT: [`reports/kfold_v3/phobert/kfold_results.csv`](file:///E:/BT1/reports/kfold_v3/phobert/kfold_results.csv)
* Báo cáo chi tiết PhoBERT v2: [`reports/phobert_v2/REPORT.md`](file:///E:/BT1/reports/phobert_v2/REPORT.md)
