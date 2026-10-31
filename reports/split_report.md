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
- Không dùng `StratifiedGroupKFold` trực tiếp vì số nhóm rất nhỏ (17 post) và fold-based split không bỏ kiểm soát được tỷ lệ 70/15/15; thuật toán trên kiểm soát cả tỷ lệ, nhóm và cân bằng nhãn.

- Random seed: `42`
- Số candidate thử: `200`
- Input: D:/Python/BT1/BT1/data/processed/cleaned.csv, output: D:/Python/BT1/BT1/data/splits/{train,val,test}.csv.

## Tổng quan dữ liệu

- Tổng số samples: 2314
- Tổng số unique posts: 17
- Tổng số toxic: 188 (8.12%)
- Size nhóm (câu mỗi post): min 13, median 113, max 432, mean 136.12

## Thống kê từng tập

| Split | Samples | Posts | Non-toxic | Toxic | Toxic % | Tỷ lệ mẫu vs mục tiêu |
|---|---:|---:|---:|---:|---:|---|
| train | 1594 | 9 | 1469 | 125 | 7.84% | 68.89% (mục tiêu 70%) |
| val | 371 | 4 | 339 | 32 | 8.63% | 16.03% (mục tiêu 15%) |
| test | 349 | 4 | 318 | 31 | 8.88% | 15.08% (mục tiêu 15%) |

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

- Số nhóm rất nhỏ (17 post); tỷ lệ 70/15/15 chỉ đạt gần (không chính xác)
  vì mỗi post có nhiều câu; ràng buộc group-aware được ưu tiên tuyệt đối.
- Dataset mất cân bằng nặng (toxic chỉ ~8%); split giữ phân bố nhãn gần
  tổng thể, không oversample/undersample hay tăng dữ liệu.
- Duplicate check thiết kế exact (lower-case `text_clean`), không fuzzy;
  câu ngắn phổ biến trùng tự nhiên cần review thủ công nếu xuất hiện.
- Test set chỉ dùng cho đánh giá cuối; không dùng để fit bất kỳ
  thành phần preprocessing có học tham số nào.
