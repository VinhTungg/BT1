# Gán nhãn bình luận PTIT cho 4 thành viên

> Bốn file annotation là phần dữ liệu gán nhãn ban đầu. Toàn bộ bộ dữ liệu hiện hành đã chốt nằm trong `data/clean/`, chọn qua `data/current_dataset.json`; xem `data/README.md`. Lịch sử nguồn được giữ cục bộ theo `archive/README.md`.

Đợt chia này gồm 2.314 bình luận từ 17 bài viết, lấy từ `data/comments.csv`.
File gốc giữ nguyên trên máy người thu thập. Bốn file dưới đây chứa đầy đủ
các cột nguồn và thêm cột `note`; mỗi bình luận được giao đúng một lần.

| Người phụ trách | File | Số bình luận |
|---|---|---:|
| Thành viên 1 | [thanh_vien_1.csv](thanh_vien_1.csv) | 579 |
| Thành viên 2 | [thanh_vien_2.csv](thanh_vien_2.csv) | 579 |
| Thành viên 3 | [thanh_vien_3.csv](thanh_vien_3.csv) | 578 |
| Thành viên 4 | [thanh_vien_4.csv](thanh_vien_4.csv) | 578 |

Nhóm tự phân công tên thật tương ứng với số thành viên. Mỗi người chỉ sửa
`label` và `note` trong file của mình. Giữ nguyên tên file, ID, nội dung,
nguồn và số dòng. File CSV dùng UTF-8 có BOM để đọc tiếng Việt trong Excel;
khi lưu chọn **CSV UTF-8**. Nếu dùng Excel, nhập qua **Data > From Text/CSV**
và đặt các cột nguồn ở dạng Text để tránh tự đổi dữ liệu. Không chạy công cụ
thu thập với `--output` trỏ vào các file gán nhãn vì có thêm cột `note`.

## Quy tắc gán nhãn

- `0`: non-toxic — trao đổi bình thường, góp ý hoặc phê bình không công kích.
- `1`: toxic — lăng mạ, hạ nhục, quấy rối, đe dọa hoặc công kích cá nhân/nhóm.
- Để trống khi chưa quyết định; ghi lý do ở `note` để cả nhóm thảo luận.
- Không tự mặc định câu tiêu cực, chê trường hoặc có từ tục đều là toxic.
- Xét mục tiêu và ngữ cảnh: câu trích lại lời xúc phạm để phản đối cần xem
  khác với câu trực tiếp xúc phạm người khác. Mỉa mai và đùa giữa bạn bè cần
  thống nhất tiêu chí; không đoán khi thiếu ngữ cảnh.
- Có thể mở `source_url` để xem ngữ cảnh bài viết. Không cần bổ sung danh tính.

Trước khi làm hết, nhóm nên chọn chung khoảng 50 ID để gán thử độc lập trong
bản sao riêng, đối chiếu và thống nhất các trường hợp bất đồng. Không ghi
nhiều nhãn vào cùng một ô trong file phân công.

## Nhận và nộp phần việc qua Git

Lần đầu tải dự án:

```powershell
git clone https://github.com/VinhTungg/BT1.git
cd BT1
```

Nếu đã có dự án, chạy `git pull --ff-only` khi thư mục làm việc sạch trước
khi bắt đầu. Khi nộp, thay số `1` bên dưới bằng số thành viên của mình:

```powershell
git add annotations/thanh_vien_1.csv
git commit -m "Label comments for member 1"
git pull --rebase
git push origin main
```

Các thành viên cần quyền ghi vào repo. Nếu không có quyền, gửi file hoàn
thành cho người quản lý repo để người đó kiểm tra rồi commit. Nếu Git báo
xung đột, giữ bản sao phần việc và nhờ người quản lý xử lý trước khi tiếp tục.

## Ghép kết quả

Sau khi đủ 4 phần, ghép theo `id`; kiểm tra đủ 2.314 ID duy nhất, các cột nguồn
khớp bản chia và nhãn chỉ là `0` hoặc `1`. Câu còn trống cần được duyệt lại.
Đừng ghép bằng cách nối văn bản thô vì bình luận có thể có xuống dòng.

Đây là phân công gán nhãn, **không phải cách chia train/validation/test**.
Khi huấn luyện cần chia theo bài viết và xử lý trùng lặp để tránh rò rỉ dữ liệu.
`manifest.json` ghi số dòng từng phần và SHA-256 của dữ liệu gốc tại thời điểm
chia. Nếu thu thập thêm, tạo đợt phân công mới; không tạo lại và ghi đè các
file mà thành viên đang gán nhãn.
