# Hướng dẫn chạy Huấn luyện PhoBERT trên Google Colab cho Người 3

Thư mục này được chuẩn bị riêng cho **Người 3 (Deep Learning PhoBERT Lead)** để đưa lên Google Colab huấn luyện mô hình ngôn ngữ `vinai/phobert-base-v2`.

---

## 📁 1. Cấu trúc thư mục

```text
colab_phobert/
├── data/
│   ├── train.csv              # Tập huấn luyện (6.000 câu, 394 toxic)
│   └── val.csv                # Tập kiểm tra (1.277 câu, 83 toxic)
├── train_phobert.ipynb        # Jupyter Notebook hoàn chỉnh từ A - Z
└── README.md                  # Tài liệu hướng dẫn này
```

> **Lưu ý về tập `test.csv`:** Theo quy chế phân công nhiệm vụ của nhóm, tập `test.csv` được **Người 4** giữ độc lập để chấm điểm công tâm ở bước cuối cùng. Người 3 chỉ sử dụng `train.csv` để học và `val.csv` để theo dõi và chọn checkpoint có Macro-F1 cao nhất.

---

## 🚀 2. Các bước chạy trên Google Colab

### Cách 1: Tải trực tiếp lên Colab (Nhanh nhất)
1. Mở trình duyệt và truy cập [Google Colab](https://colab.research.google.com/).
2. Chọn tab **Upload** (Tải lên) và chọn file `train_phobert.ipynb`.
3. Bật GPU cho Colab:
   * Trên thanh menu chọn **Runtime** (Thời gian chạy) > **Change runtime type** (Thay đổi loại thời gian chạy).
   * Chọn phần cứng: **T4 GPU** > Nhấn **Save**.
4. Ở thanh bên trái của Colab, bấm vào biểu tượng **Thư mục (Files)**:
   * Kéo thả cả folder `data` (chứa `train.csv` và `val.csv`) vào đây.
5. Nhấn **Runtime** > **Run all** (Chạy tất cả) hoặc chạy từng cell từ trên xuống dưới.

### Cách 2: Đưa lên Google Drive (Nếu muốn lưu trữ lâu dài)
1. Tải cả folder `colab_phobert` lên Google Drive của bạn.
2. Mở file `train_phobert.ipynb` bằng Google Colab.
3. Trong notebook, bỏ dấu comment ở cell mount Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. Di chuyển đường dẫn làm việc về thư mục chứa notebook trên Drive và chạy.

---

## 🎯 3. Các điểm nổi bật của Notebook này

1. **Xử lý mất cân bằng dữ liệu (Imbalanced Data)**:
   * Tỷ lệ toxic chỉ chiếm ~6.5% trong khi non-toxic chiếm ~93.5%.
   * Notebook sử dụng `compute_class_weight('balanced')` để đưa trọng số phạt vào hàm mất mát `CrossEntropyLoss(weight=class_weights)`. Điều này giúp mô hình nhận diện tốt các câu toxic và tối ưu hóa chỉ số **Macro-F1**.
2. **Tương thích hoàn hảo với dữ liệu PTIT**:
   * Sử dụng cột `text_tokenized` đã được Người 1 tách từ chuẩn bằng `underthesea` (các từ ghép nối bằng `_`).
3. **Theo dõi và lưu Checkpoint tốt nhất**:
   * Mô hình tự động lưu weights và tokenizer vào `models/phobert_best/` mỗi khi chỉ số **Macro-F1** trên tập Validation đạt đỉnh mới.
4. **Trực quan hóa và phân tích lỗi sẵn sàng cho Báo cáo BTL**:
   * Tự động sinh 2 file ảnh PNG độ nét cao: `training_curves_phobert.png` (đồ thị Loss/F1) và `confusion_matrix_phobert.png` (ma trận nhầm lẫn).
   * Xuất file CSV phân tích lỗi `validation_error_analysis.csv` liệt kê chi tiết các câu False Positive (báo nhầm) và False Negative (bỏ sót).
   * Tạo bảng so sánh hiệu năng trực tiếp với 4 Baseline và mô hình cải tiến của Người 2.

---

## 📦 4. Sản phẩm cần bàn giao cho Người 4

Sau khi chạy xong cell cuối cùng trong Notebook, chương trình sẽ tự động tạo và tải về máy file **`phobert_deliverables_nguoi_3.zip`** chứa:
1. Thư mục checkpoint tốt nhất: `models/phobert_best/` (`pytorch_model.bin` / `model.safetensors`, `config.json`, tokenizer files).
2. Thư mục báo cáo & hình ảnh: `reports/phobert/` (`training_curves_phobert.png`, `confusion_matrix_phobert.png`, `validation_error_analysis.csv`, `best_metrics.json`, `model_comparison.csv`).

Gửi file zip này cho **Người 4** để Người 4:
- Chấm điểm mô hình PhoBERT trên tập `test.csv`.
- Đưa biểu đồ và phân tích lỗi vào Chương 3 của file Báo cáo BTL và Slide thuyết trình.

