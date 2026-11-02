# Thu thập thêm dữ liệu bình luận PTIT

Bộ hiện tại đã chốt trong `data/clean/`, gồm 8.595 câu. Không chạy thu thập
để ghi đè các split này. Thống kê đợt bổ sung trước đây còn ở
`reports/data_cleanup/collection_summary.json`; dữ liệu thô, quyết định và
ảnh kiểm tra lịch sử được giữ cục bộ theo [archive/README.md](../archive/README.md).

## Tìm và thu thập bài mới

`config/groups.txt` giữ bốn nhóm đã được người dùng cho phép.
`collect_comments.py` dùng đăng nhập thủ công và profile trình duyệt riêng.
Dùng tên file mới cho mỗi đợt.

```powershell
# Chỉ tìm link bài; cần tự đăng nhập/xử lý xác minh trong trình duyệt.
.\.venv\Scripts\python.exe -X utf8 collect_comments.py --browser chrome --discover-groups config/groups.txt --urls data/incoming/post_urls_next.txt --discovery-report reports/discovery/next_candidates.csv --discovery-rounds 50 --max-posts 150 --min-comments 10 --wait-ms 1800

# Đọc các bài đã chọn, loại nội dung đã có trong ba split hiện hành.
.\.venv\Scripts\python.exe -X utf8 collect_comments.py --browser chrome --auto --urls data/incoming/post_urls_next.txt --output data/incoming/comments_next.csv --target-comments 6000 --exclude-data data/clean/train.csv --exclude-data data/clean/val.csv --exclude-data data/clean/test.csv --context-dir reports/collection_next/post_contexts --max-rounds 200 --idle-rounds 5 --wait-ms 1500
```

Chỉ đếm văn bản duy nhất thực lưu, không dùng số Facebook hiển thị thay cho
số dòng. Không tạo nội dung để bù thiếu. Dừng khi gặp giới hạn truy cập;
checkpoint/CAPTCHA phải được người dùng xử lý.

Nhãn của dữ liệu thu thập mới để trống cho đến khi nhóm chốt. Giữ ID, mã bài,
URL nguồn và thời điểm để đối chiếu. Câu cần ngữ cảnh phải được đọc đúng bài
và nhánh phản hồi. Không suy nhãn chỉ từ sự xuất hiện của từ tục.

## Đóng gói và thực nghiệm

Tạo phiên bản dataset mới sau khi nhãn hoàn tất; kiểm tra trùng ID, nội dung,
bài nguồn và mâu thuẫn nhãn. Không trộn dữ liệu mới vào test/validation đang
chốt chỉ để tăng điểm. Từ vựng và IDF phải fit lại riêng trên train mới.

`prepare_expanded_training.py` và `evaluate_expansion.py` giữ cách đóng gói
của đợt mở rộng lịch sử. Các mặc định của chúng tham chiếu cấu trúc cũ; cần
chỉ định lại đầu vào/đầu ra hoặc khôi phục snapshot trước khi tái lập đợt đó.
Chúng không phải lệnh chạy hằng ngày của pipeline hiện tại.

Để dùng mô hình đang chọn, xem [hướng dẫn bước 2–3](TEXT_FEATURE_OPTIMIZATION.md).
