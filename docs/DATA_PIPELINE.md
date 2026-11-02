# Pipeline dữ liệu hiện tại — đề 1c

Phân loại bình luận trong các nhóm Facebook liên quan sinh viên PTIT thành
`0 = non-toxic`, `1 = toxic`. Toàn bộ 8.595 nhãn hiện tại đã được nhóm chốt.

## Dữ liệu và nguồn đang dùng

`data/current_dataset.json` chọn `data/clean/manifest.json`. Không chọn bộ dữ
liệu theo một file CSV đang mở trong IDE hoặc theo các thư mục của lượt cũ.

| Tập | Số câu | Số bài | Non-toxic | Toxic |
|---|---:|---:|---:|---:|
| Train | 6.000 | 70 | 5.606 | 394 |
| Validation | 1.277 | 16 | 1.194 | 83 |
| Test | 1.318 | 22 | 1.236 | 82 |

`data/clean/all_labeled.csv` là hợp ba tập, chỉ dùng quản lý corpus. Không
fit trên toàn bộ file này rồi báo điểm validation/test.

Bốn file trong `annotations/` là phần dữ liệu gán nhãn ban đầu, không phải
các tập huấn luyện/đánh giá của toàn bộ corpus hiện tại. Thông tin theo ID và
tập nằm ở `reports/data_cleanup/provenance.csv`; thống kê đợt thu thập bổ sung
nằm ở `reports/data_cleanup/collection_summary.json`. Nguồn bài của đợt thu
thập ban đầu còn trong `post_urls.txt`, nhóm được cho phép ở `config/groups.txt`.

## Tiền xử lý và đặc trưng

Mỗi split giữ sáu cột: `id`, `post_id`, `text`, `text_clean`,
`text_tokenized`, `label`. ID, mã bài và metadata không dùng làm đặc trưng.

`src/preprocessing.py` chuẩn hóa NFC, khoảng trắng và ký tự điều khiển;
giữ dấu tiếng Việt, phủ định, dấu câu và emoji ghép; tách từ bằng underthesea.
`src/text_features.py` bổ sung chuẩn hóa viết tắt có kiểm soát, placeholder,
BoW, TF-IDF từ/ký tự và kết hợp hai nhánh.

Bộ baseline nhận `text_tokenized`. Pipeline tối ưu hiện tại nhận văn bản
thô và chứa đầy đủ bước tiền xử lý bên trong artifact. Xem
[chi tiết bước 2–3](TEXT_FEATURE_OPTIMIZATION.md).

## Kiểm tra tính toàn vẹn

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/consolidate_data.py --check
```

Lệnh chỉ kiểm tra: hash, nhãn 0/1, trường bắt buộc, chuẩn hóa, trùng ID/nội
dung, tách biệt mã bài và file gộp bằng hợp ba split. Mỗi bài nằm trọn một tập;
kiểm định chéo cũng giữ nguyên nhóm bài. Vectorizer/IDF chỉ fit trên train
hoặc phần huấn luyện của từng fold.

`audit_data.py`, `src.preprocessing` và `make_splits.py` là các công cụ tạo
một phiên bản dữ liệu mới từ annotation. Các mặc định `data/processed/` và
`data/splits/` của quy trình gốc không phải dữ liệu hiện hành. Không chạy lại
để thay bộ đã chốt khi chỉ muốn dự đoán hoặc tái lập tối ưu.

Lịch sử nguồn/đổi nhãn và đầu ra của bộ chia cũ được giữ cục bộ theo
[quy ước lưu trữ](../archive/README.md); không cần chúng để dùng bộ hiện tại.
