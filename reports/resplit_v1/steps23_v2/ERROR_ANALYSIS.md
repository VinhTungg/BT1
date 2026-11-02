# Phân tích lỗi của cấu hình đã chọn

Các dấu hiệu dưới đây chỉ dùng để mô tả lỗi, không dùng làm nhãn hay chỉnh nhãn.
OOF train đã được dùng để chọn ngưỡng; số đo này không phải đánh giá độc lập sau chọn ngưỡng.

| Tập | Dấu hiệu | Có dấu hiệu | Số câu | TP | FN | FP | Recall toxic |
|---|---|---|---:|---:|---:|---:|---:|
| train_oof_calibration | has_negation | False | 5121 | 128 | 178 | 242 | 0.4183 |
| train_oof_calibration | has_negation | True | 879 | 31 | 57 | 84 | 0.3523 |
| train_oof_calibration | has_mention | False | 2207 | 105 | 94 | 196 | 0.5276 |
| train_oof_calibration | has_mention | True | 3793 | 54 | 141 | 130 | 0.2769 |
| train_oof_calibration | short_comment | False | 4147 | 145 | 200 | 271 | 0.4203 |
| train_oof_calibration | short_comment | True | 1853 | 14 | 35 | 55 | 0.2857 |
| validation | has_negation | False | 1112 | 35 | 39 | 58 | 0.4730 |
| validation | has_negation | True | 165 | 5 | 4 | 8 | 0.5556 |
| validation | has_mention | False | 403 | 18 | 15 | 27 | 0.5455 |
| validation | has_mention | True | 874 | 22 | 28 | 39 | 0.4400 |
| validation | short_comment | False | 770 | 26 | 28 | 45 | 0.4815 |
| validation | short_comment | True | 507 | 14 | 15 | 21 | 0.4828 |

Trong train OOF, câu toxic ngắn và câu có mention bị bỏ sót nhiều hơn nhóm đối chiếu; đây là tương quan mô tả, không chứng minh nguyên nhân.
Đọc mẫu FP/FN cho thấy lỗi còn ở câu rất ngắn, cách viết biến thể, biểu đạt bóng gió và phân biệt cảm thán/phản biện với công kích. Chưa có gán loại lỗi ngữ nghĩa cho toàn bộ mẫu.
Từ dừng được bỏ ở nhánh từ, nhánh ký tự vẫn giữ nguyên nội dung. Không thêm quy tắc đặc biệt từ các lỗi validation.
Test chưa được dùng để đánh giá cấu hình mới.
