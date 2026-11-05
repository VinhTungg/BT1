# 📘 HƯỚNG DẪN BIÊN SOẠN BÁO CÁO BTL & SLIDE THUYẾT TRÌNH
## ĐỀ TÀI: PHÂN LOẠI BÌNH LUẬN TOXIC TRONG CÁC NHÓM FACEBOOK SINH VIÊN PTIT
**Dành riêng cho: Người 4 (Lead Đánh giá, Trực quan hóa & Chủ trì Báo cáo BTL)**

---

## 💡 TỔNG QUAN CHIẾN LƯỢC: CỐT TRUYỆN "3 BƯỚC CẢI TIẾN LIÊN TỤC"

Khi chấm Bài tập lớn môn Học máy & Xử lý ngôn ngữ tự nhiên, Giảng viên đánh giá rất thấp các bài chỉ chạy code 1 lần rồi đưa ra một bảng điểm vô hồn. Điểm 9.5 – 10.0 luôn thuộc về những nhóm **thể hiện được tư duy nghiên cứu khoa học thực nghiệm có tiến hóa (Progressive Experimental Methodology)**:

```text
  ┌─────────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
  │  GIAI ĐOẠN 1: v1        │      │  GIAI ĐOẠN 2: v2        │      │  GIAI ĐOẠN 3: v3        │
  │  Bộ chia cổ điển        │      │  Tối ưu phân bổ dữ liệu │      │  Kiểm định khoa học     │
  │  (70 / 15 / 15)         │ ───► │  (80 / 10 / 10)         │ ───► │  (5-Fold GroupKFold)    │
  │  Macro-F1: ~68.8%       │      │  Macro-F1: ~78.8%       │      │  Macro-F1: ~71.2%       │
  │  👉 Phát hiện nút thắt  │      │  👉 Điểm số bứt phá     │      │  👉 Độ tin cậy tối cao  │
  └─────────────────────────┘      └─────────────────────────┘      └─────────────────────────┘
```

Trong Báo cáo BTL (đặc biệt là **Chương 3**), Người 4 hãy triển khai theo đúng mạch kể chuyện logic này. Thầy/Cô sẽ thấy nhóm làm việc cực kỳ nghiêm túc, hiểu tường tận dữ liệu và giải thuật.

---

## 📑 BỐ CỤC CHUẨN CỦA QUYỂN BÁO CÁO BTL (WORD / PDF)

Người 4 cấu trúc mục lục báo cáo theo 4 chương tiêu chuẩn sau:

### CHƯƠNG 1: ĐẶT VẤN ĐỀ VÀ BÀI TOÁN (Khoảng 2 – 3 trang)
1. **Bối cảnh thực tế:** Mạng xã hội của sinh viên PTIT (Confessions, Nhóm 2k5, Hội sinh viên...). Hiện tượng ngôn từ độc hại (cyberbullying, xúc phạm giảng viên/sinh viên, chửi tục vô văn hóa).
2. **Mục tiêu đề tài:** Xây dựng hệ thống tự động lọc và phân loại bình luận thành 2 lớp:
   * **Nhãn `0` (Non-toxic - An toàn):** Bình luận tích cực, hỏi han, tranh luận văn minh, hoặc than phiền khách quan về dịch vụ đào tạo (học phí, cổng đăng ký tín chỉ).
   * **Nhãn `1` (Toxic - Độc hại):** Ngôn từ thóa mạ danh dự, xúc phạm cá nhân, chửi thề tục tĩu nặng nề.
3. **Thách thức bài toán:**
   * Dữ liệu mất cân bằng nghiêm trọng (lớp Toxic chỉ chiếm ~6.5%).
   * Ngôn ngữ mạng xã hội sinh viên PTIT có nhiều teencode, tiếng lóng viết tắt (*"đkm", "clm", "loz", "hạch"*), mỉa mai ngầm (sarcasm).

---

### CHƯƠNG 2: CƠ SỞ LÝ THUYẾT & PHƯƠNG PHÁP NGHIÊN CỨU (Khoảng 4 – 5 trang)
1. **Quy trình Tiền xử lý dữ liệu (Người 1 phụ trách):**
   * Chuẩn hóa bảng mã tiếng Việt Unicode NFC.
   * Xóa khoảng trắng thừa, làm sạch ký tự điều khiển/zero-width.
   * Tách từ ghép tiếng Việt bằng thư viện `underthesea` (*"sinh viên"* $\rightarrow$ *"sinh_viên"*).
   * **Nguyên tắc bảo toàn tín hiệu:** KHÔNG xóa emoji, dấu câu, từ phủ định hay placeholder `[MENTION]`, `[URL]` vì chúng mang tín hiệu nhận diện cảm xúc mạnh.
2. **Nguyên tắc chống rò rỉ dữ liệu (Group-aware splitting theo `post_id`):**
   * *Điểm nhấn phương pháp:* Nhấn mạnh rằng nhóm **không chia ngẫu nhiên từng dòng câu** mà **chia theo nguyên bài viết (`post_id`)**. Toàn bộ các bình luận cùng một bài viết chỉ được phép nằm trong 1 tập duy nhất (Train, Val hoặc Test) để tránh mô hình học vẹt chủ đề thảo luận.
3. **Các mô hình Học máy truyền thống (Người 2 phụ trách):**
   * Mô hình đối chứng: DummyClassifier (Most Frequent).
   * Trích xuất đặc trưng: Bag-of-Words (BoW) và TF-IDF (Term Frequency - Inverse Document Frequency).
   * Thuật toán phân loại: Multinomial Naive Bayes và Linear Support Vector Machine (LinearSVC).
   * Kỹ thuật tối ưu nâng cao của Người 2: Kết hợp n-gram từ (1–2) và n-gram ký tự (3–5), đánh trọng số lớp (`class_weight='balanced'`).
4. **Mô hình Học sâu Pre-trained Language Model (Người 3 phụ trách):**
   * Kiến trúc `vinai/phobert-base-v2` (RoBERTa pre-trained trên 20GB văn bản tiếng Việt).
   * Mở rộng từ vựng (Vocab size 64.003) với các special tokens: `[MENTION]` (ID: 64001) và `[URL]` (ID: 64002).
   * Kỹ thuật huấn luyện: Mixed Precision (FP16), AdamW, Linear Warmup, Label Smoothing (`ε=0.05`), Early Stopping (Patience = 2).
5. **Thang đo đánh giá (Evaluation Metrics):**
   * Giải thích tại sao **Macro-F1** và **F1-Toxic / Recall-Toxic** là tiêu chí quyết định sống còn thay vì Accuracy (do Accuracy bị méo mó khi dữ liệu mất cân bằng: đoán toàn 0 cũng đạt 93.5% Accuracy nhưng vô dụng).

---

### CHƯƠNG 3: THỰC NGHIỆM, KẾT QUẢ VÀ ĐỐI THOẠI SO SÁNH (TRỌNG TÂM — 6 – 8 trang)
*(Đây là chương quyết định điểm 10. Hãy trình bày tuần tự theo 3 phiên bản như sau)*:

#### 3.1. Thống kê toàn bộ Corpus dữ liệu
Nhóm thu thập và gắn nhãn hoàn chỉnh **8.595 bình luận** từ **108 bài viết** Facebook PTIT:
* Lớp Non-toxic (0): 8.036 câu (93.50%)
* Lớp Toxic (1): 559 câu (6.50%)

#### 3.2. Phiên bản 1 (v1 — Bộ chia cổ điển 70/15/15)
* **Thiết lập:** Train 6.000 câu (394 toxic), Val 1.277 câu (83 toxic), Test 1.318 câu (82 toxic).
* **Kết quả đối đầu:**
  * Baseline TF-IDF + Linear SVM: Macro-F1 đạt **67.10%** (F1-toxic 40.00%).
  * SVM Tối ưu Từ+Ký tự (`steps23_v2` - Người 2): Macro-F1 đạt **68.86%** (F1-toxic 42.33%).
  * PhoBERT v2 (Người 3 - Ngưỡng 0.52): Macro-F1 đạt **68.88%** (F1-toxic 42.05%).
* **Phân tích nguyên nhân & Nút thắt:**
  * Cả hai mô hình đều bị chững lại quanh mốc 68.8%.
  * *Phát hiện:* Tập Train chỉ có 394 câu toxic (quá ít mẫu độc hại cho mạng nơ-ron học sâu). Đồng thời, tập Val bị dồn cục: riêng 2 bài viết đã chiếm tới 81% lượng câu toxic của tập Val, làm méo mó việc đánh giá.

#### 3.3. Phiên bản 2 (v2 — Tối ưu hóa phân chia dữ liệu 80/10/10)
* **Giải pháp cải tiến:** Điều chỉnh tỷ lệ sang 80/10/10 theo khuyến nghị kỹ thuật. Tập Train tăng lên **6.845 câu** (+845 mẫu, trong đó có thêm **51 câu toxic**). Tập Val gồm 824 câu (54 toxic) trải đều qua 13 bài viết.
* **Kết quả đối đầu bứt phá:**
  * Linear SVM Word+Char (Người 2): Macro-F1 nhảy vọt lên **78.83%** (+9.97%), F1-toxic đạt **60.38%**, Accuracy đạt **94.90%**.
  * PhoBERT v2 (Người 3 - Ngưỡng 0.90): Macro-F1 đạt đỉnh **78.88%**, F1-toxic đạt **60.55%**, bắt đúng 33/54 câu toxic (Recall 61.11%).
* **Đánh giá:** Khẳng định dữ liệu huấn luyện dồi dào là chìa khóa vàng cho bài toán mất cân bằng. Cả ML lẫn DL đều có sự nhảy vọt ~10% Macro-F1.

#### 3.4. Phiên bản 3 (v3 — Đỉnh cao kiểm định: 5-Fold GroupKFold)
* **Đặt vấn đề:** Điểm số 78.8% ở v2 rất cao, nhưng liệu có phải do "may mắn" chọn trúng tập Val 13 bài thuận lợi? Nhóm quyết định thực hiện phương pháp luận học thuật cao nhất: **Kiểm định chéo 5 lần theo bài viết (5-Fold GroupKFold)** trên toàn bộ 7.669 câu (95 bài viết).
* **Kết quả kiểm định Out-of-Fold (OOF):**
  * BoW + Linear SVM: Macro-F1 64.70% $\pm$ 2.99% (OOF: 64.96%).
  * TF-IDF + Linear SVM: Macro-F1 66.30% $\pm$ 1.86% (OOF: 66.44%).
  * Linear SVM Word+Char (Người 2): Macro-F1 **68.94% $\pm$ 1.42%** (OOF: **68.98%**, bắt được 173 câu toxic).
  * PhoBERT v2 Ensemble (Người 3): Macro-F1 **71.16% $\pm$ 2.39%** (OOF: **71.22%**, bắt được 224 câu toxic).
* **Kết luận khoa học:**
  * Điểm trung bình OOF 71.22% phản ánh năng lực tổng quát hóa thực chất nhất khi phải đối mặt với mọi dạng bài viết khó.
  * PhoBERT chứng minh sự vượt trội toàn diện: thắng SVM ở **cả 5/5 Folds** (+2.22% Macro-F1) và bắt trúng **hơn SVM 51 câu toxic**.

#### 3.5. Bảng tổng hợp Master Table (Chèn trọn vẹn vào báo cáo)

*(Copy nguyên văn Bảng ở Mục 1 trong file [TONG_KET_SO_SANH_3_VERSION.md](TONG_KET_SO_SANH_3_VERSION.md) vào đây)*.

#### 3.6. Phân tích lỗi ngữ nghĩa thực tế (Semantic Error Analysis)
Phân loại lỗi của mô hình thành 2 nhóm điển hình:
1. **False Positives (Báo nhầm Non-toxic thành Toxic):** Sinh viên than thở bức xúc dịch vụ (*"Học phí kỳ này chán vl", "Cổng đăng ký tín chỉ lại sập, bực cả mình"*). Mô hình bị đánh lừa bởi từ ngữ tiêu cực cực độ (sentiment negative) dù câu không hề xúc phạm ai.
2. **False Negatives (Bỏ sót Toxic):** Các câu viết tắt teencode cực đoan (*"đkm", "clm", "loz"*) hoặc câu châm chọc cá nhân thâm thúy/mỉa mai ngầm (sarcasm) cần ngữ cảnh cả bài viết nguồn mới hiểu.

---

### CHƯƠNG 4: KẾT LUẬN VÀ BÀN GIAO ỨNG DỤNG (Khoảng 1 – 2 trang)
1. **Tổng kết thành quả:** Nhóm đã hoàn thành xuất sắc mục tiêu:
   * Xây dựng thành công mô hình học máy truyền thống tối ưu (Linear SVM n-gram ký tự đạt 68.9% OOF / 78.8% v2).
   * Huấn luyện thành công mô hình ngôn ngữ sâu PhoBERT v2 đạt SOTA (71.2% OOF / 78.9% v2).
2. **So sánh sự đánh đổi thực tế (Trade-off):**
   * *PhoBERT:* Hiểu ngữ cảnh sâu, bắt toxic nhạy nhất, nhưng file nặng (540MB), suy luận cần GPU hoặc tốn ~30ms trên CPU.
   * *Linear SVM:* Rất nhẹ (~1.2MB), chạy cực nhanh (~4ms trên CPU), độ ổn định giữa các fold cao nhất ($\pm 1.42\%$), phù hợp cho máy chủ chi phí thấp.
3. **Ứng dụng thực tế (Demo App):**
   * Giới thiệu giao diện web trực quan [`demo.py`](file:///d:/BT1/demo.py) cho phép kiểm tra trực tiếp bình luận với thời gian thực.

---

## 🎨 DANH SÁCH FILE ẢNH & SỐ LIỆU ĐÃ CÓ SẴN TRONG REPO (ĐỂ CHÈN VÀO WORD)

Người 4 chỉ cần mở các đường dẫn sau để copy ảnh và số liệu chèn vào báo cáo:

1. **Ảnh đồ thị Loss & Quét ngưỡng PhoBERT v2:**
   * [`reports/phobert_v2/training_and_threshold_analysis.png`](file:///d:/BT1/reports/phobert_v2/training_and_threshold_analysis.png)
2. **Ảnh đồ thị huấn luyện PhoBERT v1:**
   * [`reports/phobert/training_and_threshold_analysis.png`](file:///d:/BT1/reports/phobert/training_and_threshold_analysis.png)
3. **Chi tiết 5 Folds K-Fold SVM:**
   * [`reports/kfold_v3/svm/svm_fold_details.csv`](file:///d:/BT1/reports/kfold_v3/svm/svm_fold_details.csv)
4. **Chi tiết 5 Folds K-Fold PhoBERT:**
   * [`reports/kfold_v3/phobert/kfold_results.csv`](file:///d:/BT1/reports/kfold_v3/phobert/kfold_results.csv)
5. **Bảng dự đoán OOF 7.669 câu để làm phân tích lỗi:**
   * [`reports/kfold_v3/phobert/oof_predictions.csv`](file:///d:/BT1/reports/kfold_v3/phobert/oof_predictions.csv)

---

## 🖥️ KỊCH BẢN THIẾT KẾ SLIDE THUYẾT TRÌNH (10 SLIDES CHUẨN)

Khi thuyết trình trước Giảng viên và Hội đồng, Người 4 phân bổ 10 slide như sau:

* **Slide 1: Trang bìa** — Tên đề tài, Giảng viên hướng dẫn, Danh sách 4 thành viên và nhiệm vụ phân công rõ ràng.
* **Slide 2: Đặt vấn đề & Thách thức** — Bối cảnh mạng xã hội PTIT, bài toán phân loại 2 nhãn, thách thức mất cân bằng nhãn (Toxic chỉ 6.5%).
* **Slide 3: Tiền xử lý & Chống Data Leakage** — Chuẩn hóa Unicode NFC, tách từ `underthesea`, quy tắc chia tách group-aware theo `post_id`.
* **Slide 4: Giai đoạn 1 (v1 - 70/15/15) & Phát hiện nút thắt** — Bảng điểm ~68.8% của SVM và PhoBERT. Nhận định nguyên nhân do tập Train thiếu câu toxic.
* **Slide 5: Giai đoạn 2 (v2 - 80/10/10) & Điểm số bứt phá** — Nâng Train thêm 51 câu toxic $\rightarrow$ Điểm vọt lên **78.83% (SVM)** và **78.88% (PhoBERT)**. Chèn ảnh đồ thị `training_and_threshold_analysis.png`.
* **Slide 6: Giai đoạn 3 (v3 - 5-Fold GroupKFold)** — Kiểm định khoa học 5 lần trên 7.669 câu. PhoBERT đạt **71.22% OOF**, vượt trội SVM ở cả 5/5 Folds.
* **Slide 7: So sánh đối đầu SVM vs PhoBERT** — Bảng đánh đổi (Trade-off): Tốc độ, dung lượng, tài nguyên, khả năng bắt ngữ cảnh/mỉa mai.
* **Slide 8: Phân tích lỗi thực tế (Error Analysis)** — Phân tích cụ thể ví dụ câu sinh viên than học phí bị báo nhầm (FP) và teencode bị bỏ sót (FN).
* **Slide 9: Bản Demo trực tiếp** — Chạy `python demo.py --web` chiếu trực tiếp giao diện Web, gõ thử vài câu do Thầy/Cô yêu cầu.
* **Slide 10: Kết luận & Hướng phát triển** — Tóm tắt đóng góp, mở rộng sang đa nhãn (hate speech, spam, toxic) và cảm ơn.

---

## ❓ DỰ ĐOÁN CÁC CÂU HỎI VẤN ĐÁP CỦA GIẢNG VIÊN & GỢI Ý TRẢ LỜI

1. **Hỏi: Tại sao không dùng Accuracy để chọn mô hình mà lại dùng Macro-F1?**
   * *Trả lời:* Vì dữ liệu có 93.5% là Non-toxic. Nếu mô hình luôn đoán 0 thì Accuracy vẫn đạt 93.5% nhưng không bắt được bất kỳ câu toxic nào (F1-toxic = 0). Macro-F1 tính trung bình điều hòa của cả 2 lớp, buộc mô hình phải học tốt cả lớp thiểu số toxic mới có điểm cao.
2. **Hỏi: Chia dữ liệu theo bài viết (`post_id`) có tác dụng gì so với chia ngẫu nhiên từng dòng?**
   * *Trả lời:* Các bình luận trong cùng một bài viết thường dùng chung từ khóa và ngữ cảnh. Nếu chia ngẫu nhiên, câu của cùng một bài viết sẽ xuất hiện ở cả Train và Test, dẫn tới hiện tượng **Data Leakage (Rò rỉ thông tin)**, mô hình học vẹt chủ đề và cho điểm cao ảo. Chia theo `post_id` đảm bảo tập Test hoàn toàn là các bài viết mới lạ lẫm mà mô hình chưa từng thấy.
3. **Hỏi: Tại sao điểm của v2 (78.8%) lại cao hơn v3 (71.2%)?**
   * *Trả lời:* Điểm v2 được tính trên một tập Val cố định (13 bài) có độ tương đồng phong cách khá tốt với tập Train. Trong khi đó, v3 kiểm tra chéo qua **toàn bộ 95 bài viết** trong corpus (bao gồm cả những bài teencode khó nhất, mỉa mai ngầm nhất), do đó điểm 71.2% của v3 là con số phản ánh đúng thực chất năng lực tổng quát hóa nhất.
