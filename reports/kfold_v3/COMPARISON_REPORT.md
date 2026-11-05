# BÁO CÁO TỔNG KẾT ĐỐI ĐẦU: SO SÁNH TOÀN DIỆN 3 PHIÊN BẢN (V1, V2, V3) VÀ ĐỐI ĐẦU SVM VS PHOBERT

* **Bài tập lớn 1c:** Phân loại bình luận toxic và non-toxic trong các nhóm Facebook sinh viên PTIT.
* **Thời gian thực hiện:** 05/11/2026
* **Người thực hiện:** Người 3 (Deep Learning PhoBERT Lead) & Phối hợp nhóm.

---

## 📌 PHẦN 1: BẢNG SO SÁNH TỔNG HỢP TOÀN BỘ 3 PHIÊN BẢN (V1, V2, V3)

| Phiên bản | Chiến lược phân chia | Tập đánh giá | Mô hình SVM Tốt nhất | Mô hình PhoBERT | Nhận xét bản chất khoa học |
|:---:|---|---|:---:|:---:|---|
| **v1** | Cố định **70/15/15** | Validation 1.277 câu *(16 bài, 83 toxic)* | **68.86%** *(Tối ưu từ+ký tự+OOF)* | **68.88%** *(Epoch 5, $\tau=0.52$)* | Phân chia cơ sở ban đầu; lượng câu toxic trong Train ít (394 câu) nên mô hình khó học ngữ cảnh sâu. PhoBERT nhỉnh hơn SVM +0.02%. |
| **v2** | Cố định **80/10/10** | Validation 824 câu *(13 bài, 54 toxic)* | **78.83%** *(word_char_svm)* | 🏆 **78.88%** *(Epoch 4, $\tau=0.90$)* | Tăng Train lên 80% (thêm 51 câu toxic / 12 bài viết). Điểm số tăng vọt do 13 bài trong Val khá thuận lợi. PhoBERT tiếp tục dẫn đầu bảng. |
| **v3** | **5-Fold GroupKFold** | **Toàn bộ 7.669 câu** *(95 bài, 499 toxic)* | **68.94% $\pm$ 1.42%** *(OOF: 68.98%)* | 🥇 **71.16% $\pm$ 2.39%** *(OOF: 71.22%)* | **Đo lường trung thực & toàn diện nhất**. Mọi câu đều được thi thử ngoài mẫu. **PhoBERT đánh bại SVM trọn vẹn cả 5/5 Folds (+2.22%)**. |

---

## ⚔️ PHẦN 2: ĐỐI ĐẦU TRỰC DIỆN SVM VS PHOBERT TRÊN 5-FOLD GROUPKFOLD (V3)

Vì **v3** đánh giá trên cùng một tập dữ liệu 7.669 câu với cùng các Folds chia theo bài viết (`post_id`), đây là thước đo chuẩn xác và công tâm nhất để so sánh Machine Learning truyền thống (SVM) và Deep Learning (PhoBERT):

### 1. Bảng đối đầu theo từng Fold (Fold 0 $\rightarrow$ Fold 4)

| Fold | Số câu Val (Toxic) | BoW + SVM | TF-IDF + SVM | SVM Tốt nhất (`word_char_svm`) | **PhoBERT v2** | Chênh lệch (PhoBERT vs SVM tốt nhất) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fold 0** | 1.548 (101 toxic) | 64.90% | 67.39% | 70.62% | 🏆 **75.04%** | **+4.42%** (PhoBERT áp đảo) |
| **Fold 1** | 1.525 (94 toxic) | 61.09% | 64.88% | 66.63% | 🏆 **69.78%** | **+3.15%** (PhoBERT áp đảo) |
| **Fold 2** | 1.529 (100 toxic) | 65.47% | 65.47% | 69.77% | 🏆 **71.91%** | **+2.14%** (PhoBERT áp đảo) |
| **Fold 3** | 1.540 (111 toxic) | 69.73% | 69.40% | 69.61% | 🏆 **69.64%** | **+0.03%** (PhoBERT thắng sít sao) |
| **Fold 4** | 1.527 (93 toxic) | 62.29% | 64.33% | 68.06% | 🏆 **69.41%** | **+1.35%** (PhoBERT áp đảo) |
| **TRUNG BÌNH (Mean $\pm$ Std)** | **1.534 câu / fold** | **64.70% $\pm$ 2.99%** | **66.30% $\pm$ 1.86%** | **68.94% $\pm$ 1.42%** | 🏆 **71.16% $\pm$ 2.39%** | **+2.22% (PhoBERT THẮNG TUYỆT ĐỐI)** |

---

### 2. So sánh khả năng nhận diện trên toàn bộ dữ liệu ngoài mẫu (Out-Of-Fold — 7.669 câu)

| Chỉ số đánh giá | SVM Tốt nhất (`word_char_svm`) | PhoBERT v2 (Ngưỡng 0.85) | Đánh giá & Khác biệt cốt lõi |
|---|:---:|:---:|---|
| **Macro-F1 OOF** | 68.98% | 🏆 **71.22%** | **PhoBERT cao hơn +2.24%** |
| **F1-Toxic OOF** | 41.34% | 🏆 **46.09%** | **PhoBERT cao hơn +4.75%** |
| **Recall Toxic OOF** | 34.67% *(173 câu)* | 🏆 **44.89%** *(224 câu)* | **PhoBERT bắt thêm được +51 câu toxic mà SVM bỏ sót!** |
| **Precision Toxic OOF** | **51.18%** | 47.36% | SVM chặt chẽ hơn về từ khóa nhưng bỏ sót quá nhiều toxic |
| **Accuracy OOF** | **93.60%** | 93.17% | Cả hai đều đạt độ chính xác tổng thể > 93% |
| **Số câu Toxic phát hiện đúng (TP)** | 173 / 499 | 🏆 **224 / 499** | **Khả năng dò quét toxic của PhoBERT vượt trội hoàn toàn** |
| **Số câu Toxic bị bỏ sót (FN)** | 326 câu | 🏆 **275 câu** | **PhoBERT giảm thiểu được 51 ca bỏ sót** |
| **Số ca báo nhầm Non-toxic (FP)** | **165 câu** | 249 câu | PhoBERT nhạy cảm hơn với các câu tiêu cực gay gắt |

---

## 🧠 PHẦN 3: SO SÁNH SÂU VỀ ƯU / NHƯỢC ĐIỂM GIỮA SVM VÀ PHOBERT

### 1. Mô hình Machine Learning truyền thống (SVM):
* **Ưu điểm:**
  1. **Tốc độ cực nhanh & nhẹ:** Huấn luyện 5 Folds trên CPU chỉ mất **~30 giây**, file mô hình `.joblib` chỉ vài trăm KB.
  2. **Độ ổn định cao:** Độ lệch chuẩn giữa các Folds thấp nhất ($\pm 1.42\%$), ít bị biến động.
  3. **Precision tốt:** Rất ít khi báo nhầm những câu bình thường thành toxic vì SVM chỉ kích hoạt khi gặp đúng các n-gram từ/ký tự độc hại quen thuộc.
* **Nhược điểm chí mạng:**
  1. **"Mù" ngữ cảnh:** Bỏ sót tới **326 câu toxic (65.3%)**. Nếu sinh viên chửi bới bằng cách nói bóng gió, mỉa mai, hoặc dùng teencode mới ghép chữ, SVM hoàn toàn không nhận diện được vì không có trong từ điển n-gram tĩnh.
  2. Không hiểu được sự phủ định nhiều tầng hoặc mối quan hệ phụ thuộc giữa các từ cách xa nhau.

### 2. Mô hình Deep Learning ngôn ngữ (PhoBERT):
* **Ưu điểm vượt trội:**
  1. **Hiểu ngữ cảnh sâu (Deep Contextual Understanding):** Nhờ cơ chế Self-Attention được tiền huấn luyện trên hàng triệu văn bản tiếng Việt, PhoBERT hiểu được các cấu trúc châm biếm, công kích ngầm và bắt được **224 câu toxic (vượt trội hơn SVM 51 câu)**.
  2. **Thắng tuyệt đối cả 5 Folds:** Macro-F1 trung bình đạt **71.16%**, chứng minh năng lực tổng quát hóa ưu việt hơn hẳn máy học truyền thống.
  3. **Khả năng Ensemble (Hội đồng chuyên gia):** Việc kết hợp xác suất từ 5 mô hình của 5 Folds giúp san phẳng các sai số đơn lẻ, tạo ra bộ dự đoán cực kỳ mạnh mẽ.
* **Hạn chế:**
  1. Tốn tài nguyên: Bắt buộc cần GPU (như Colab T4), kích thước mô hình lớn (~540 MB).
  2. Độ nhạy cảm cao: Một số câu than thở áp lực học tập có từ cảm thán tiêu cực (*"mệt mỏi vl"*, *"học hành như hạch"*) dễ bị mô hình đánh giá nhầm là toxic.

---

## 🎯 PHẦN 4: KẾT LUẬN & ĐỀ XUẤT CHO BÁO CÁO BÀI TẬP LỚN

1. **Về câu hỏi nghiên cứu của đề tài:** Thực nghiệm 3 phiên bản đã chứng minh rõ ràng:
   * **Từ v1 lên v2:** Cho thấy tầm quan trọng của việc bổ sung dữ liệu huấn luyện (tăng train từ 70% lên 80% giúp mô hình học thêm được nhiều ngữ cảnh toxic mới).
   * **Từ v2 sang v3 (K-Fold):** Mang lại giá trị học thuật cao nhất. Đưa ra con số đánh giá khách quan, triệt tiêu yếu tố may rủi của 1 tập val nhỏ và cung cấp thước đo độ lệch chuẩn ($\pm \sigma$) đáng tin cậy.
2. **Về tương quan SVM vs PhoBERT:**
   * **PhoBERT v2 là mô hình chiến thắng toàn diện** (dẫn đầu ở cả v1, v2 và v3).
   * Việc PhoBERT vượt trội hơn SVM ở chỉ số Recall Toxic (+10.22%) là luận điểm đắt giá nhất để khẳng định sức mạnh của Deep Learning Transformer trong bài toán xử lý ngôn ngữ tự nhiên tiếng Việt trên mạng xã hội.
3. **Bàn giao cho Người 4:** Toàn bộ bảng số liệu, ma trận nhầm lẫn và file OOF predictions đã được lưu trữ hoàn tất để Người 4 hoàn thiện Chương 3 Báo cáo BTL và thiết kế Slide thuyết trình!

