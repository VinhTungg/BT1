# Báo cáo Audit dữ liệu

- Tổng số file annotation tìm thấy: 4
- Tổng bản ghi trước xử lý: 2314
- Bản ghi hợp lệ: 2314
- Bản ghi lỗi (loại có log): 0
- Duplicate bị loại (có log): 0
- Conflicting label (không sửa, chỉ báo cáo): 0
- Trùng văn bản khác id (chỉ báo cáo): 0
- Tổng dòng trong `data_issues.csv`: 0
- Conservation check (hợp lệ + lỗi + trùng = tổng): PASS

## Số bản ghi từng file

| File | Số bản ghi |
|---|---:|
| thanh_vien_1.csv | 579 |
| thanh_vien_2.csv | 579 |
| thanh_vien_3.csv | 578 |
| thanh_vien_4.csv | 578 |

## Phân bố nhãn (trên bản ghi hợp lệ)

| Nhãn | Số lượng | Tỷ lệ |
|---|---:|---:|
| 0 (non-toxic) | 2126 | 91.88% |
| 1 (toxic) | 188 | 8.12% |

- Số lượng `post_id` duy nhất: 17
- Số câu trung bình mỗi bài viết: 136.12

## Độ dài văn bản (trên bản ghi hợp lệ)

| Chỉ số | Ký tự | Từ |
|---|---:|---:|
| mean | 60.89 | 13.84 |
| median | 36.0 | 8.0 |
| min | 1 | 1 |
| max | 1507 | 340 |

## Chính sách xử lý

- Bản ghi label trống/không thuộc {0,1}, text/post_id/id trống bị loại khỏi `merged.csv`, có truy vết trong `data_issues.csv`.
- Trùng `id` hoặc trùng toàn bộ bản ghi: giữ bản ghi đầu tiên, log đầy đủ.
- Cùng văn bản khác id hoặc khác nhãn: KHÔNG tự loại, KHÔNG sửa nhãn, chỉ báo cáo.
- Nhiều câu cùng `post_id` là hành vi hợp lệ, không được coi là duplicate.
- File annotation gốc không bị sửa.
