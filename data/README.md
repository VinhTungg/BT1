# Dữ liệu chính

Toàn bộ dữ liệu đang dùng nằm trong **`data/clean/`**. Đây là bộ **`resplit_v2` (tỷ lệ 80/10/10)**
được tối ưu từ toàn bộ corpus 8.595 câu đã gộp. Bộ cũ `resplit_v1` (70/15/15) được lưu trữ an toàn tại `archive/before_resplit_v2_20261104_094708/`. Đường dẫn mặc định được chọn qua
[current_dataset.json](current_dataset.json).

| File | Mục đích | Số câu | Non-toxic | Toxic | Tỷ lệ Toxic |
|---|---|---:|---:|---:|---:|
| [clean/all_labeled.csv](clean/all_labeled.csv) | Toàn bộ corpus đã gộp, dùng quản lý dữ liệu | 8.595 | 8.036 | 559 | 6.50% |
| [clean/train.csv](clean/train.csv) | Huấn luyện (~80%) | 6.845 | 6.400 | 445 | 6.50% |
| [clean/val.csv](clean/val.csv) | Chọn mô hình và tối ưu ngưỡng (~10%) | 824 | 770 | 54 | 6.55% |
| [clean/test.csv](clean/test.csv) | Đánh giá cuối (~10%) | 926 | 866 | 60 | 6.48% |
| [clean/manifest.json](clean/manifest.json) | Mã băm, thống kê và chính sách đánh giá | | | | |

`all_labeled.csv` là hợp của ba tập, không phải dữ liệu bổ sung. **Không dùng toàn bộ
file này để fit mô hình rồi báo điểm trên val/test**, vì các câu đánh giá đã nằm trong đó.
Mỗi bài viết thuộc trọn một tập (chống data leakage): train 82 bài, validation 13 bài, test 13 bài.

## Làm sạch và loại dư thừa

Đã kiểm tra trường bắt buộc, nhãn 0/1, Unicode NFC, ký tự ẩn, khoảng trắng,
trùng ID, trùng nội dung sau chuẩn hóa và đầu vào tách từ. Không trùng bài viết
giữa các tập, kể cả khác tiền tố nhưng cùng số ID bài Facebook.

Bộ này đã loại 3 câu trùng ở bước tạo `resplit_v1`; lượt gom này không phát hiện
câu trùng cần bỏ thêm. Nội dung và nhãn của 8.595 câu cùng byte của ba file split
được giữ nguyên. Không lọc theo độ dài, không xóa từ tục, phủ định, emoji hay dấu câu.

Giữ sáu cột để tương thích pipeline và đối chiếu: `id`, `post_id`, `text`,
`text_clean`, `text_tokenized`, `label`. Baseline chỉ học từ `text_tokenized`;
ID và thông tin nguồn không dùng làm đặc trưng. Các cột nguồn, người duyệt và lý do
không lặp lại trong file huấn luyện.

## Nguồn nhãn và lịch sử

Toàn bộ **8.595 nhãn đã được nhóm chốt làm bộ nhãn chuẩn** ngày 02/11/2026.
Giữ nguyên giá trị nhãn, ID và phân chia train/validation/test.

Đối chiếu nguồn theo ID và tập tại [provenance.csv](../reports/data_cleanup/provenance.csv).
Kết quả kiểm tra tại [audit.json](../reports/data_cleanup/audit.json).
Bốn file trong `annotations/` vẫn giữ nguyên để nhóm làm việc.

Các bộ chia và snapshot trước đây được giữ trong [lịch sử cục bộ](../archive/README.md).
Repository chỉ chứa dữ liệu hiện hành và các artifact cần để tái lập thực nghiệm.
Các đường dẫn nguồn trước khi lưu trữ trong metadata là thông tin lịch sử.

## Kiểm tra và sử dụng

Từ thư mục gốc dự án:

```powershell
# Chỉ đọc và kiểm tra; không ghi lại hoặc chia lại dữ liệu
.\.venv\Scripts\python.exe -X utf8 scripts/consolidate_data.py --check

# Các lệnh train mặc định tự đọc data/clean/ qua current_dataset.json
.\.venv\Scripts\python.exe -X utf8 scripts/train_baselines.py

# Dùng mô hình đã có, không cần huấn luyện lại sau khi đổi đường dẫn
.\.venv\Scripts\python.exe -X utf8 scripts/predict_current.py "Cảm ơn bạn đã chia sẻ"
```

Bốn baseline được giữ làm mốc so sánh. Mô hình hiện tại nhận văn bản thô,
đã tối ưu tiền xử lý và đặc trưng trên cùng train/validation; xem
[báo cáo hiện tại](../reports/resplit_v1/steps23_v2/REPORT.md).
