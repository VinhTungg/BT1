# Lịch sử cục bộ

Các bản dữ liệu nguồn, quyết định nhãn, ảnh kiểm tra và snapshot trước những
lần thay đổi được giữ cục bộ trong thư mục này, không đưa lên Git. Repository
chứa đầy đủ dữ liệu đã chốt, thông tin truy vết theo ID, các mô hình cần cho
thực nghiệm hiện tại và báo cáo để chạy/kiểm tra mà không cần giải nén lịch sử.

Lượt dọn ngày 02/11/2026 nén các đầu ra thuộc bộ chia cũ thành
`legacy_outputs_20261102_162553.zip`, rồi xóa bản rời khỏi thư mục làm việc.
Đã đối chiếu SHA-256 từng tệp trong bản nén trước khi xóa. File `MANIFEST.json`
bên trong bản nén ghi tên đường dẫn cũ, kích thước và SHA-256 để khôi phục.

Danh sách dọn nằm trong `reports/cleanup_20261102.json`. Các đường dẫn
`source_file_before_archive` và snapshot trong metadata là thông tin lịch sử,
không phải đầu vào huấn luyện hiện tại.

Ba lượt mô hình còn dùng nằm trong `models/resplit_v1/`: `baselines`,
`steps23_v1` và `steps23_v2`. Mô hình hiện tại được chọn bởi
`models/current_run.json`; bộ dữ liệu nằm trong `data/clean/`.
