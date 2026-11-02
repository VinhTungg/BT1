# Tối ưu có kiểm định theo bài viết

Giữ nguyên nhãn và bộ chia. Ngưỡng chỉ học từ dự đoán ngoài mẫu trên train.
Tám cấu hình có kiểm soát: hai cách chuẩn hóa × ngram từ 1–2/1–3 × giữ/bỏ danh sách từ dừng nhỏ.
Đặc trưng ký tự giữ nguyên; chỉ bỏ từ dừng ở nhánh từ. Không bỏ phủ định hoặc đại từ chỉ người.

| Mô hình | Ngưỡng | Macro-F1 val | F1-toxic val | Recall-toxic val | Macro-F1 fold ± std (ngưỡng 0) |
|---|---:|---:|---:|---:|---|
| reference_slang_word_char_svm_zero | 0.0000 | 0.6851 | 0.4153 | 0.4578 | 0.6554 ± 0.0123 |
| reference_slang_word_char_svm_oof | 0.0000 | 0.6851 | 0.4153 | 0.4578 | 0.6554 ± 0.0123 |
| reference_preserve_bow_svm_zero | 0.0000 | 0.6846 | 0.4167 | 0.4819 | 0.6212 ± 0.0224 |
| reference_preserve_bow_svm_oof | 0.1822 | 0.6600 | 0.3613 | 0.3373 | 0.6212 ± 0.0224 |
| reference_slang_tfidf_svm_zero | 0.0000 | 0.6819 | 0.4196 | 0.5663 | 0.6352 ± 0.0211 |
| reference_slang_tfidf_svm_oof | 0.1736 | 0.6849 | 0.4118 | 0.4217 | 0.6352 ± 0.0211 |
| preserve_ng2_keep_zero | 0.0000 | 0.6805 | 0.4066 | 0.4458 | 0.6564 ± 0.0119 |
| preserve_ng2_keep_oof | -0.0269 | 0.6790 | 0.4062 | 0.4699 | 0.6564 ± 0.0119 |
| preserve_ng2_stop_zero | 0.0000 | 0.6811 | 0.4086 | 0.4578 | 0.6592 ± 0.0160 |
| preserve_ng2_stop_oof | 0.0158 | 0.6773 | 0.4000 | 0.4337 | 0.6592 ± 0.0160 |
| preserve_ng3_keep_zero | 0.0000 | 0.6824 | 0.4108 | 0.4578 | 0.6581 ± 0.0141 |
| preserve_ng3_keep_oof | 0.0094 | 0.6805 | 0.4066 | 0.4458 | 0.6581 ± 0.0141 |
| preserve_ng3_stop_zero | 0.0000 | 0.6837 | 0.4130 | 0.4578 | 0.6583 ± 0.0158 |
| preserve_ng3_stop_oof | 0.0111 | 0.6805 | 0.4066 | 0.4458 | 0.6583 ± 0.0158 |
| slang_ng2_keep_zero | 0.0000 | 0.6851 | 0.4153 | 0.4578 | 0.6554 ± 0.0123 |
| slang_ng2_keep_oof | 0.0000 | 0.6851 | 0.4153 | 0.4578 | 0.6554 ± 0.0123 |
| slang_ng2_stop_zero | 0.0000 | 0.6811 | 0.4086 | 0.4578 | 0.6545 ± 0.0134 |
| slang_ng2_stop_oof | -0.0026 | 0.6811 | 0.4086 | 0.4578 | 0.6545 ± 0.0134 |
| slang_ng3_keep_zero | 0.0000 | 0.6869 | 0.4194 | 0.4699 | 0.6539 ± 0.0122 |
| slang_ng3_keep_oof | 0.0078 | 0.6837 | 0.4130 | 0.4578 | 0.6539 ± 0.0122 |
| slang_ng3_stop_zero | 0.0000 | 0.6855 | 0.4171 | 0.4699 | 0.6546 ± 0.0114 |
| slang_ng3_stop_oof | -0.0030 | 0.6886 | 0.4233 | 0.4819 | 0.6546 ± 0.0114 |

Tốt nhất theo validation: **slang_ng3_stop_oof**.

Configurations partly originate from previous validation selection; OOF diagnostics and threshold calibration are not unbiased nested-CV estimates.
Bảng OOF có gắn dấu hiệu phủ định, mention và câu ngắn để phân tích lỗi; các dấu hiệu này không được dùng làm nhãn.
Chưa đọc test; kết quả test chỉ xuất sau khi khóa lựa chọn.

## Quyết định sử dụng

Đã xác minh nạp lại và tái lập toàn bộ 1277 dự đoán validation cho cả mô hình cũ và mô hình được chọn.
Macro-F1: 0.685086 → 0.688596; F1-toxic: 0.415301 → 0.423280; Recall-toxic: 0.457831 → 0.481928.
TP 38 → 40; FN 45 → 43; FP 62 → 66.
Mức tăng validation nhỏ; kiểm định theo bài viết chưa chứng minh mức cải thiện vượt trội. Giữ kết quả thử nghiệm không tốt hơn để đối chiếu.
Mô hình hiện tại được cập nhật qua `models/current_run.json`. Toàn bộ ID, nhãn, byte train/validation/test giữ nguyên.
Chi tiết lỗi: [ERROR_ANALYSIS.md](ERROR_ANALYSIS.md), `error_analysis.csv`, `error_groups.csv`.
Theo phạm vi hiện tại chỉ tối ưu mô hình truyền thống; chưa huấn luyện PhoBERT và chưa đánh giá test cho cấu hình mới.
