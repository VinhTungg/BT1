# Hướng Dẫn Chạy Huấn Luyện PhoBERT 5-Fold GroupKFold & Ensemble (V3)

Thư mục này được chuẩn bị riêng cho **Người 3 (Deep Learning PhoBERT Lead)** để triển khai chiến lược **5-Fold Cross Validation theo nhóm bài viết (`post_id`)** và mô hình kết hợp **Ensemble** trên Google Colab.

---

## 📁 1. Cấu trúc thư mục

```text
colab_phobert_kfold/
├── data/
│   └── train_pool_5folds.csv    # Toàn bộ 7.669 câu của tập huấn luyện đã gán sẵn cột `fold` (0 đến 4)
├── train_phobert_kfold.ipynb    # Jupyter Notebook hoàn chỉnh từ A - Z
└── README.md                    # Tài liệu hướng dẫn này
```

> **Lưu ý về tập `test.csv`:** Tập kiểm thử cuối cùng `test.csv` (926 câu) vẫn được bảo toàn nguyên vẹn cho **Người 4**. Mô hình Ensemble 5 Folds sau khi huấn luyện xong sẽ là công cụ mạnh nhất để bàn giao cho Người 4 chấm điểm thi!

---

## 🚀 2. Cách chạy trên Google Colab

1. **Tải lên Colab:**
   * Mở [Google Colab](https://colab.research.google.com/).
   * Upload file `train_phobert_kfold.ipynb`.
2. **Bật GPU:**
   * Chọn menu **Runtime (Thời gian chạy)** > **Change runtime type (Thay đổi loại thời gian chạy)** > Chọn **T4 GPU** > Nhấn **Lưu**.
3. **Nạp dữ liệu:**
   * Ở cột bên trái, bấm vào biểu tượng **📁 Files**.
   * Kéo thả thư mục `data` (chứa `train_pool_5folds.csv`) vào khung Files của Colab.
4. **Chạy huấn luyện:**
   * Nhấn **Runtime** > **Run all (Chạy tất cả)**.
   * Quá trình huấn luyện 5 Folds (mỗi Fold 4 Epochs) sẽ diễn ra tuần tự trong khoảng **12 – 15 phút** trên GPU T4.

---

## 🌟 3. Điểm vượt trội của chiến lược V3

1. **Đo lường độ ổn định thực chất:** Báo cáo ghi nhận cả giá trị trung bình và độ lệch chuẩn **Mean Macro-F1 $\pm$ Std** qua 5 Folds, triệt tiêu hoàn toàn yếu tố may rủi.
2. **Dự đoán toàn diện Out-of-Fold (OOF):** Toàn bộ 7.669 câu trong tập huấn luyện đều có dự đoán OOF khách quan để Người 4 phân tích ma trận nhầm lẫn và lỗi sai.
3. **Ensemble 5 Mô Hình:** Khi chạy Bước 8, hệ thống tự động kết hợp 5 mô hình của 5 Folds để dự đoán bằng cách lấy trung bình cộng xác suất, giúp tăng vọt độ chính xác khi đối đầu với các câu bình luận toxic khó.
4. **Tự động đóng gói bàn giao:** Bước 9 tự động nén toàn bộ 5 checkpoints và báo cáo thành file `phobert_kfold_deliverables_nguoi_3.zip` tải về máy tính.

