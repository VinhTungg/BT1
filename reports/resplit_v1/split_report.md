# Báo cáo Split dữ liệu

## Phương pháp splitting

- **Group-aware splitting theo `post_id`**: mỗi nhóm (các câu cùng bài
  viết) được gán nguyên vẹn vào một tập duy nhất, chống data leakage.
- Thuật toán: thử giới hạn hữu hạn candidate hoán vị nhóm (mỗi candidate
  dùng `random.Random(seed + i)`), gán nhóm bằng chiến lược
  *largest-remaining-bin* theo tỷ lệ 70/15/15, chấm điểm cost theo:
  (1) sai lệch tỷ lệ số mẫu so với mục tiêu;
  (2) sai lệch tỷ lệ toxic giữa tập với toàn dataset (trọng số 5);
  (3) hình phạt lớn nếu tập nào thiếu hẳn một lớp.
- Chọn candidate có cost thấp nhất; tie-break theo thứ tự candidate →
  kết quả deterministic, reproducible với cùng seed.
- Chia trực tiếp ba tập 70/15/15 bằng thuật toán nhóm của dự án;
  các fold đồng đều của cross-validation phục vụ một mục đích khác.

- Random seed: `42`
- Số candidate thử: `200`
- Input: data/experiments/resplit_v1/all_labeled.csv, output: data/experiments/resplit_v1/{train,val,test}.csv.

## Tổng quan dữ liệu

- Tổng số samples: 8595
- Tổng số unique posts: 108
- Tổng số toxic: 559 (6.50%)
- Size nhóm (câu mỗi post): min 8, median 61, max 431, mean 79.58

## Thống kê từng tập

| Split | Samples | Posts | Non-toxic | Toxic | Toxic % | Tỷ lệ mẫu vs mục tiêu |
|---|---:|---:|---:|---:|---:|---|
| train | 6000 | 70 | 5606 | 394 | 6.57% | 69.81% (mục tiêu 70%) |
| val | 1277 | 16 | 1194 | 83 | 6.50% | 14.86% (mục tiêu 15%) |
| test | 1318 | 22 | 1236 | 82 | 6.22% | 15.33% (mục tiêu 15%) |

## Kiểm tra chống leakage & bảo toàn dữ liệu

- Train/Val post overlap: PASS
- Train/Test post overlap: PASS
- Val/Test post overlap: PASS
- Tổng số bản ghi các tập = đầu vào: PASS
- Multiset `id` bảo toàn (không mất/sinh bản ghi): PASS
- Không có `id` trùng trong split outputs: PASS

## Duplicate text bất thường giữa các tập

- Không phát hiện exact duplicate text giữa train/val/test.

## Giới hạn còn tồn tại

- Có 108 nhóm bài viết; tỷ lệ 70/15/15 chỉ đạt gần (không chính xác)
  vì mỗi post có nhiều câu; ràng buộc group-aware được ưu tiên tuyệt đối.
- Toxic chiếm 6.50%; split giữ phân bố nhãn gần
  tổng thể, không oversample/undersample hay tăng dữ liệu.
- Duplicate check thiết kế exact (lower-case `text_clean`), không fuzzy;
  câu ngắn phổ biến trùng tự nhiên cần review thủ công nếu xuất hiện.
- Test set chỉ dùng cho đánh giá cuối; không dùng để fit bất kỳ
  thành phần preprocessing có học tham số nào.
