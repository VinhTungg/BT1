# Kết quả gom và kiểm tra dữ liệu

Bộ chính nằm tại `data/clean/`, tổng 8.595 câu: 8.036 non-toxic và 559 toxic.
`all_labeled.csv` là hợp của train 6.000, val 1.277, test 1.318, chỉ dùng để quản lý corpus.
Các lệnh huấn luyện mặc định đã trỏ tới thư mục mới.

Đã kiểm tra chuẩn hóa NFC/khoảng trắng/ký tự ẩn, dữ liệu bắt buộc, nhãn 0/1,
ID trùng, nội dung chuẩn hóa trùng, đầu vào tách từ trùng và bài viết dùng chung.
Không có lỗi trong các kiểm tra này. Lượt gom này bỏ thêm 0 câu; 3 câu trùng
đã được loại khi tạo resplit_v1 trước đó.

Giữ nguyên từng byte của ba split và file gộp. Artifact mô hình không đổi,
không cần huấn luyện lại do chuyển đường dẫn. Đã xác nhận đường dẫn mặc định
cho baseline/cải tiến, hash của năm artifact và lệnh dự đoán hiện tại.
24 kiểm thử liên quan đều đạt.

Thư mục data trước đó có 13 CSV thuộc các lượt khác nhau; hiện chỉ có 4 CSV
chính. Bản cũ nằm trong `archive/data_before_consolidation_20261101_024306/`.
Đây là dọn thư mục làm việc và giữ bản lưu, không phải giải phóng dung lượng ổ đĩa.
Các báo cáo và manifest lịch sử giữ đường dẫn tại thời điểm tạo; xem relocation.json
trong bản lưu để tìm lại hoặc khôi phục dữ liệu cũ.

Toàn bộ 8.595 nhãn đã được nhóm chốt ngày 02/11/2026; không sửa giá trị nhãn.
Nguồn từng ID và phân chia nằm tại [provenance.csv](provenance.csv).

Chạy lại kiểm tra chỉ đọc:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/consolidate_data.py --check
```

Hai trường hợp biên đã có kiểm thử: câu toxic ngắn/emoji được giữ lại;
cùng nội dung nhưng khác nhãn bị từ chối để tránh tự chọn nhãn.
