> Dữ liệu đã được gom sang `data/clean/`, giữ nguyên nhãn và phân chia.
> Manifest hiện tại: [data/clean/manifest.json](../../data/clean/manifest.json).
> Hồ sơ cũ nằm trong `archive/data_before_consolidation_20261101_024306`.

# Đánh giá cách chia dữ liệu và bản đề xuất resplit_v1

**Cập nhật:** Người dùng đã chọn dùng bản này. Dataset đã kích hoạt trong
`data/current_dataset.json`, bốn baseline đã được huấn luyện lại và đánh giá.
Kết quả: [TRAINING_REPORT.md](TRAINING_REPORT.md). Các phần đề xuất bên dưới
ghi lại phân tích tại thời điểm tạo bộ chia.

## Hiện trạng

Bản dùng để huấn luyện gần nhất có 7.878 train, 371 validation và 349 test (91,63% / 4,31% / 4,06%). Các tập không chung bài viết; giữ hai tập cũ giúp so sánh mô hình cũ/mới trên cùng benchmark.

Train có 100 bài và 468 toxic (5,94%); validation có 4 bài và 47 toxic (12,67%); test có 4 bài và 44 toxic (12,61%). Một bài chứa 35/47 câu toxic của validation và một bài chứa 27/44 câu toxic của test. Kết quả vì vậy phụ thuộc nhiều vào ít bài; không thể kết luận toàn bộ lỗi mô hình do cách chia.

## Bản đề xuất riêng

Gộp 8.598 câu có nhãn từ train mở rộng + validation/test hiện tại. Loại ba dòng non-toxic trùng sau chuẩn hóa chữ thường/khoảng trắng, còn 8.595 câu từ 108 bài, gồm 559 toxic. Không sửa nhãn, không dùng 140 câu chưa rõ nhãn.

Giữ cả bài trong cùng tập; seed 42, 200 candidate theo tiêu chí số mẫu và phân bố nhãn. Không chạy mô hình để chọn seed.

| Tập | Số câu | Số bài | Toxic | Tỷ lệ toxic |
|---|---:|---:|---:|---:|
| train  |  6000  |  70  |  394  |  6.57% |
| val  |  1277  |  16  |  83  |  6.50% |
| test  |  1318  |  22  |  82  |  6.22% |

Các kiểm tra đều đạt: bảo toàn toàn bộ ID sau lọc; không chung ID, ID số bài viết, text_clean hoặc text_tokenized giữa các tập. 15 kiểm thử hiện có của bộ chia đều đạt. File hiện tại, mô hình đã train và kết quả đánh giá cũ giữ nguyên mã băm.

## Cách sử dụng và giới hạn

- Bản đề xuất đã được chấp nhận và trở thành bộ đang dùng (`active`). Số đo nằm trong `TRAINING_REPORT.md`; mô tả dưới đây giữ nguyên các giới hạn phương pháp.
- Nếu dùng bản này, phải huấn luyện lại từ đầu cả vectorizer và classifier; mô hình hiện tại đã học nhiều câu nay chuyển sang test đề xuất. Không dùng nó để đo khả năng trên dữ liệu chưa thấy của split mới.
- Toàn bộ pool đã xuất hiện trong quá trình phát triển trước đây. Chia lại không tạo ra một benchmark hoàn toàn độc lập với lịch sử dự án; muốn xác nhận độc lập cần bài viết mới và nhãn kiểm định độc lập.
- Toàn bộ nhãn đã được nhóm chốt làm bộ nhãn chuẩn ngày 02/11/2026.
- Không so trực tiếp điểm mới/cũ để kết luận mô hình cải thiện khi tập test đã đổi. Chốt split trước, tune trên train/validation, rồi đánh giá test. Không đổi seed để chọn điểm mô hình cao.

## Tệp

- Dữ liệu: `data/experiments/resplit_v1/{train,val,test}.csv`.
- Pool sau lọc: `data/experiments/resplit_v1/all_labeled.csv`.
- Ba bản ghi loại trùng: `data/experiments/resplit_v1/excluded_duplicates.csv`.
- Metadata/hash: `data/experiments/resplit_v1/manifest.json`.
- Chi tiết thuật toán và checks: `reports/resplit_v1/split_report.json` và `split_report.md`.

## Kiểm tra lại

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_make_splits.py -v
```

Trường hợp biên: bài rất lớn vẫn phải nằm trọn một tập dù lệch tỷ lệ mẫu; phát hiện hai nhãn khác nhau cho cùng nội dung phải dừng để rà nhãn, không tự bỏ phiếu.

Tham khảo phương pháp: https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data
