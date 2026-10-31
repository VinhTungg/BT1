# Phân Công Nhiệm Vụ Bài Tập Lớn 1c
## Đề tài: Phân loại bình luận toxic và non-toxic trong các nhóm Facebook sinh viên PTIT

* **Quy mô nhóm:** 4 thành viên
* **Phương thức gán nhãn:** **4 thành viên tự phụ trách gán nhãn/kiểm duyệt độc lập file của mình**, sau đó gửi lại cho Người 1 để gộp và xử lý.
* **Quy ước nhãn:**
  * `0` — **Non-toxic:** Trao đổi bình thường, hỏi đáp, phê bình/góp ý chính sách, dịch vụ trường lớp không có yếu tố xúc phạm, mạ lỵ cá nhân/tập thể.
  * `1` — **Toxic:** Lăng mạ, sỉ nhục, quấy rối, đe dọa, xúc phạm danh dự cá nhân hoặc nhóm người.

---

## 🔄 QUY TRÌNH PHỐI HỢP DỮ LIỆU GIỮA 4 NGƯỜI

```text
       [4 thành viên nhận 4 file trong thư mục annotations/]
                               │
   ┌───────────────────────────┼───────────────────────────┐
   ▼                           ▼                           ▼                           ▼
👤 Thành viên 1             👤 Thành viên 2             👤 Thành viên 3             👤 Thành viên 4
(Tự duyệt file 1)           (Tự duyệt file 2)           (Tự duyệt file 3)           (Tự duyệt file 4)
   │                           │                           │                           │
   └───────────────────────────┼───────────────────────────┘
                               │
           (4 thành viên gửi file đã hoàn thành cho Người 1)
                               │
                               ▼
                        👤 Người 1
         1. Gộp 4 file thành 1 bộ dữ liệu tổng (2.314 dòng)
         2. Chạy Preprocessing: chuẩn hóa Unicode, tách từ underthesea
         3. Chia tập theo post_id thành: train, val, test
                               │
         ┌─────────────────────┴─────────────────────┐
         ▼                                           ▼
 [train.csv (70%) + val.csv (15%)]               [test.csv (15%)]
         │                                           │
         ├──────────────────┐                        │
         ▼                  ▼                        │
   👤 Người 2          👤 Người 3                    │
 (Train 4 Baseline)  (Train PhoBERT)                 │
         │                  │                        │
         └─────────┬────────┘                        │
                   ▼                                 ▼
            [Bàn giao Models] ────────────────> 👤 Người 4
                                         (Chấm điểm thi trên test.csv,
                                          vẽ biểu đồ Dashboard & viết Báo cáo)
```

---

## 👤 Người 1: Data, Preprocessing & Split Lead
*(Phụ trách duyệt file 1, nhận file từ nhóm, gộp dữ liệu, chạy tiền xử lý và chia tập)*

### Các đầu việc chính:
1. **Tự gán nhãn / duyệt file của mình:**
   * Tự hoàn thành việc gán nhãn cho file [annotations/thanh_vien_1.csv](file:///d:/BT1/annotations/thanh_vien_1.csv) (579 câu).
2. **Nhận file & Gộp dữ liệu:**
   * Thu nhận 3 file đã gán nhãn xong từ Người 2, Người 3 và Người 4.
   * Viết script `audit_data.py`: Gộp 4 file thành 1 bảng dữ liệu tổng duy nhất (2.314 dòng), kiểm tra không còn dòng trống nhãn, phát hiện dữ liệu lỗi/trùng.
3. **Chạy Tiền xử lý (Preprocessing):**
   * Viết module và chạy tiền xử lý `src/preprocessing.py` trên toàn bộ văn bản:
     * Chuẩn hóa bảng mã tiếng Việt (Unicode NFC).
     * Xóa khoảng trắng thừa, làm sạch ký tự rác.
     * **Tách từ ghép tiếng Việt** (ví dụ: *"sinh viên"* $\rightarrow$ *"sinh_viên"*, *"học phí"* $\rightarrow$ *"học_phí"* bằng thư viện `underthesea`).
4. **Chia tập dữ liệu (Split) & Bàn giao:**
   * Viết script `make_splits.py`: Cắt bảng dữ liệu đã làm sạch thành 3 file **theo nguyên bài viết (`post_id`)** để chống rò rỉ dữ liệu:
     * `data/splits/train.csv` (~70%, ~1.600 câu): **Bàn giao cho Người 2 và Người 3**.
     * `data/splits/val.csv` (~15%, ~350 câu): **Bàn giao cho Người 2 và Người 3**.
     * `data/splits/test.csv` (~15%, ~350 câu): **Bàn giao cho Người 4**.

### Sản phẩm bàn giao:
* File nhãn hoàn chỉnh `annotations/thanh_vien_1.csv`
* `scripts/audit_data.py` (script gộp & kiểm tra dữ liệu)
* Module tiền xử lý `src/preprocessing.py`
* `scripts/make_splits.py` và 3 file dữ liệu sạch: `train.csv`, `val.csv`, `test.csv`

---

## 👤 Người 2: Machine Learning Baseline Lead
*(Phụ trách duyệt file 2 và huấn luyện các mô hình học máy truyền thống)*

### Các đầu việc chính:
1. **Tự gán nhãn / duyệt file của mình:**
   * Tự hoàn thành việc gán nhãn cho file [annotations/thanh_vien_2.csv](file:///d:/BT1/annotations/thanh_vien_2.csv) (579 câu) rồi gửi lại cho Người 1.
2. **Trích xuất đặc trưng văn bản:**
   * Nhận `train.csv` và `val.csv` từ Người 1 (dữ liệu này đã được Người 1 làm sạch và tách từ sẵn).
   * Cài đặt 2 phương pháp biểu diễn đặc trưng:
     * **Bag-of-Words (BoW):** Dùng `CountVectorizer` của scikit-learn.
     * **TF-IDF:** Dùng `TfidfVectorizer` của scikit-learn.
3. **Huấn luyện mô hình:**
   * Xây dựng pipeline scikit-learn huấn luyện 4 tổ hợp mô hình:
     1. BoW + Multinomial Naive Bayes (`MultinomialNB`)
     2. BoW + Linear SVM (`LinearSVC`)
     3. TF-IDF + Multinomial Naive Bayes (`MultinomialNB`)
     4. TF-IDF + Linear SVM (`LinearSVC`)
     *(Bổ sung 1 mô hình đối chứng `DummyClassifier` luôn đoán lớp đa số)*.
   * Dùng tập `val.csv` để tinh chỉnh siêu tham số (`alpha`, `C`, `ngram_range`) và chọn ra cấu hình tốt nhất của từng mô hình.
   * Lưu các file mô hình tốt nhất (`.joblib`) vào thư mục `models/` và bàn giao cho Người 4.

### Sản phẩm bàn giao:
* File nhãn hoàn chỉnh `annotations/thanh_vien_2.csv`
* `scripts/train_baselines.py`
* Thư mục `models/` chứa các file pipeline Baseline `.joblib`
* Bảng tham số tối ưu và kết quả trên tập val bàn giao cho Người 4

---

## 👤 Người 3: Deep Learning PhoBERT Lead
*(Phụ trách duyệt file 3 và huấn luyện mô hình ngôn ngữ PhoBERT)*

### Các đầu việc chính:
1. **Tự gán nhãn / duyệt file của mình:**
   * Tự hoàn thành việc gán nhãn cho file [annotations/thanh_vien_3.csv](file:///d:/BT1/annotations/thanh_vien_3.csv) (578 câu) rồi gửi lại cho Người 1.
2. **Thiết lập & Huấn luyện PhoBERT:**
   * Nhận `train.csv` và `val.csv` từ Người 1.
   * Thiết lập môi trường PyTorch + Transformers trên máy có GPU (hoặc tạo notebook trên Google Colab / Kaggle).
   * Xây dựng Dataset, DataLoader và Tokenizer cho `vinai/phobert-base`.
   * Viết pipeline fine-tune PhoBERT với đầu phân loại 2 lớp (Toxic / Non-toxic).
   * Dùng tập `val.csv` để theo dõi loss và validation metric qua từng epoch, lưu lại checkpoint mô hình đạt Macro-F1 cao nhất.
   * Bàn giao checkpoint mô hình tốt nhất cho Người 4.

### Sản phẩm bàn giao:
* File nhãn hoàn chỉnh `annotations/thanh_vien_3.csv`
* `scripts/train_phobert.py` (hoặc `notebooks/train_phobert.ipynb`)
* Thư mục checkpoint mô hình PhoBERT tốt nhất (`models/phobert_best/`)
* Bảng log quá trình train bàn giao cho Người 4

---

## 👤 Người 4: Evaluation, Visualization & Report Lead
*(Phụ trách duyệt file 4, đánh giá trên tập Test, trực quan hóa biểu đồ và chủ trì viết Báo cáo BTL)*

### Các đầu việc chính:
1. **Tự gán nhãn / duyệt file của mình:**
   * Tự hoàn thành việc gán nhãn cho file [annotations/thanh_vien_4.csv](file:///d:/BT1/annotations/thanh_vien_4.csv) (578 câu) rồi gửi lại cho Người 1.
2. **Đánh giá & Chấm điểm mô hình trên tập Test:**
   * Nhận file `test.csv` từ Người 1 và giữ độc lập (không đưa cho Người 2, 3 lúc họ đang train).
   * Nhận các mô hình đã chốt từ Người 2 (4 Baseline) và Người 3 (PhoBERT).
   * Viết script `evaluate_models.py` chạy chấm điểm tất cả mô hình trên tập `test.csv`.
   * Tính toán các chỉ số: **Accuracy, Precision, Recall, F1 toxic, Macro-F1**.
   * Lập bảng so sánh đối đầu giữa các phương pháp: BoW vs TF-IDF, Naive Bayes vs SVM vs PhoBERT.
   * Lọc các câu đoán sai (False Positive, False Negative) để làm bảng **Phân tích lỗi (Error Analysis)**.
3. **Xây dựng Dashboard Trực quan hóa (Visualization Dashboard):**
   * Xây dựng giao diện Streamlit gọn nhẹ (`app.py`) hoặc script xuất ảnh tự động để **sinh biểu đồ phục vụ Báo cáo**:
     * Biểu đồ tròn tỷ lệ nhãn Toxic / Non-toxic, biểu đồ phân bố bình luận theo bài viết.
     * Biểu đồ cột so sánh Macro-F1 và F1-toxic giữa 5 mô hình.
     * Heatmap Ma trận nhầm lẫn (Confusion Matrix) của từng mô hình (hiển thị rõ TP, FP, TN, FN).
     * Ô nhập câu nhanh để demo tại chỗ nếu giảng viên yêu cầu.
4. **Chủ trì biên soạn Báo cáo BTL & Slide:**
   * Viết file Báo cáo BTL (Word / PDF) hoàn chỉnh:
     * **Chương 1:** Đặt vấn đề, bối cảnh Facebook PTIT, bài toán và định nghĩa nhãn.
     * **Chương 2:** Cơ sở lý thuyết (BoW, TF-IDF, Naive Bayes, Linear SVM, PhoBERT, các chỉ số P/R/F1).
     * **Chương 3:** Thực nghiệm (mô tả dữ liệu, bảng so sánh kết quả 5 mô hình, chèn các biểu đồ và ma trận nhầm lẫn từ Dashboard, phân tích lỗi sai thực tế).
     * **Chương 4:** Kết luận, các hạn chế và hướng phát triển tương lai.
   * Thiết kế Slide thuyết trình (8–12 slide tóm tắt các điểm nổi bật của đề tài).

### Sản phẩm bàn giao:
* File nhãn hoàn chỉnh `annotations/thanh_vien_4.csv`
* `scripts/evaluate_models.py`
* Dashboard trực quan hóa `app.py`
* Thư mục `results/` (chứa các file ảnh biểu đồ PNG độ nét cao, `confusion_matrix_*.png`, `error_analysis.csv`)
* **File Báo cáo BTL hoàn chỉnh (Word / PDF)**
* **File Slide thuyết trình (PowerPoint / PDF)**
