# Tiền xử lý và trích đặc trưng cho đề 1c

Bộ nhãn hiện tại đã được nhóm chốt ngày 02/11/2026. Thực nghiệm giữ nguyên
8.595 ID, nhãn và phân chia: 6.000 train, 1.277 validation, 1.318 test.
Đọc dữ liệu đang chọn qua `data/current_dataset.json`.

## Lượt hiện tại: kiểm định theo bài viết và ngưỡng SVM

Kết quả hiện tại: [steps23_v2/REPORT.md](../reports/resplit_v1/steps23_v2/REPORT.md).
Lượt này chỉ tối ưu mô hình truyền thống, chưa huấn luyện PhoBERT.

`scripts/refine_text_models.py` thực hiện ba cấu hình tham chiếu và tám biến
thể cố định. Ba fold từ 70 bài train giữ nguyên toàn bộ bình luận của từng
bài; từ vựng, min_df và IDF được fit lại riêng trên phần huấn luyện mỗi fold.

Tám biến thể là hai cấu hình chuẩn hóa × từ 1–2/1–3 × giữ/bỏ danh sách từ
dừng nhỏ. Từ dừng chỉ áp dụng ở nhánh từ, còn nhánh ký tự giữ nguyên văn bản.
Danh sách: `và, là, của, ở, tại, trong, trên, dưới, theo`.

Mỗi cấu hình được đánh giá với ngưỡng 0 và ngưỡng học từ dự đoán ngoài mẫu
trên train. Không dùng validation để tính ngưỡng; validation chọn giữa các
ứng viên đã cố định. Có 22 ứng viên cho 11 cấu hình đặc trưng. Không đọc test
trong script tối ưu này.

Cấu hình được chọn: `slang_ng3_stop_oof`, Linear SVM C=0,1/balanced;
TF-IDF từ 1–3, ký tự 3–5, min_df=2, trọng số hai nhánh bằng nhau;
ngưỡng **-0,002950521489135968**. Macro-F1 validation **68,86%**,
F1-toxic **42,33%**, Recall-toxic **48,19%**.

Mức tăng Macro-F1 so với lượt trước khoảng 0,35 điểm phần trăm; chưa có bằng
chứng về một mức cải thiện lớn. Trung bình Macro-F1 ba fold ở ngưỡng 0 của
cấu hình này là 65,46%, độ lệch chuẩn 1,14 điểm phần trăm. Các cấu hình tham
chiếu từng được lựa chọn trên validation và OOF được dùng để chọn ngưỡng,
nên đây là chẩn đoán độ ổn định/hiệu chỉnh, không phải nested CV độc lập.

```powershell
# Dùng tên thư mục mới để giữ nguyên lượt đã chốt.
.\.venv\Scripts\python.exe -X utf8 scripts/refine_text_models.py --model-dir models/resplit_v1/steps23_v2_repeat --report-dir reports/resplit_v1/steps23_v2_repeat
```

Script ghi ứng viên và báo cáo vào thư mục mới; không tự đổi con trỏ mô hình.
Lượt `steps23_v2` đã được xác minh và chọn trong `models/current_run.json`;
quyết định và bản con trỏ trước đó nằm ở `selection_record.json` và
`previous_current_run.json` trong thư mục báo cáo của lượt này.

`train_fold_assignments.csv`, `group_cv_folds.csv`, `threshold_search.csv`,
`oof_predictions_*.csv` lưu quá trình kiểm định và hiệu chỉnh.
`ERROR_ANALYSIS.md`, `error_analysis.csv`, `error_groups.csv` mô tả các lỗi
còn lại mà không thay đổi nhãn. Các phần dưới ghi chi tiết lượt `steps23_v1`.

## Bước 2: tiền xử lý

`src/preprocessing.py` chuẩn hóa Unicode NFC, chuyển tab/xuống dòng thành
khoảng trắng, loại ký tự điều khiển còn lại và gom khoảng trắng. Ký tự nối
được giữ trong emoji ghép; emoji ghép không bị bộ tách từ chia thành các
biểu tượng riêng. Chuẩn hóa NFC được thực hiện sau khi loại ký tự ẩn để
các dấu tổ hợp vẫn được ghép đúng.

`src/text_features.py` có hai cấu hình để so sánh:

- `preserve`: giữ từ phủ định, từ tục, đại từ, dấu câu và emoji; quy URL về
  `[URL]`, giữ `[MENTION]`/`[URL]` thành token riêng trước khi tách từ.
- `slang`: áp dụng thêm từ điển cố định cho `ko`, `k0`, `khum`, `hok`,
  `khong`, `đc`, `thik`, `cx`, `rùi`, `nma`. Chỉ thay token đầy đủ, phân biệt
  hoa/thường. Không thay `k`, `r`, `đk`, `100k`, `2k5` hay từ chứa các chuỗi này.

Không xóa stop words hàng loạt hoặc dùng stemming tiếng Anh cho tiếng Việt.
Bộ trích đặc trưng chuyển về chữ thường. Các quy tắc viết tắt là giả định
thực nghiệm; chỉ chọn nếu tốt hơn trên validation.

Chuẩn hóa có thể khiến hai câu khác nhau như `ko biết` và `không biết`
có cùng biểu diễn. Script báo số va chạm này; không gộp dòng, đổi nhãn,
loại câu đánh giá hay học quy tắc từ validation. Kiểm tra ID, bài nguồn và
nội dung gốc được thực hiện trước khi áp dụng quy tắc.

## Bước 3: trích đặc trưng

Giữ BoW và TF-IDF để đáp ứng yêu cầu đề. Thử thêm TF-IDF ký tự và TF-IDF
kết hợp từ/ký tự để xử lý cách viết biến thể. Đặc trưng ký tự là nhánh mở
rộng, không thay thế bốn tổ hợp BoW/TF-IDF với NB/SVM.

| Nhánh | Cấu hình đặc trưng được thử |
|---|---|
| BoW | unigram hoặc unigram+bigram; min_df=1 hoặc 2 |
| TF-IDF từ | cùng ngram/min_df với BoW; sublinear_tf=True, smooth_idf=True, norm=l2 |
| TF-IDF ký tự | char_wb, ngram 2–5 hoặc 3–5; min_df=2 |
| Từ + ký tự | từ 1–2, ký tự 3–5; min_df=2; trọng số ký tự 0,5 hoặc 1 |

NB thử alpha=0,1 hoặc 1. SVM tuyến tính thử C=0,1/1/10 với trọng số lớp
balanced và ngưỡng quyết định mặc định 0. Tổng cộng 104 cấu hình cố định
cho hai cấu hình tiền xử lý; không tìm kiếm trên test.

Từ vựng, min_df và IDF chỉ học trên train. Validation chỉ được transform.
`controlled_comparison.csv` giữ unigram, min_df=1 và tham số classifier cố
định để so sánh tiền xử lý/BoW/TF-IDF. Bảng mô hình tốt nhất mỗi nhánh có thể
dùng cấu hình khác nhau nên không thay thế bảng so sánh có kiểm soát này.

## Chạy và đọc kết quả

```powershell
# Thư mục đích phải chưa tồn tại. Dùng tên mới cho mỗi lượt thực nghiệm.
.\.venv\Scripts\python.exe -X utf8 scripts/optimize_text_features.py --model-dir models/resplit_v1/steps23_repeat --report-dir reports/resplit_v1/steps23_repeat --no-promote

# Dự đoán bằng mô hình được chọn hiện tại; nhập trực tiếp văn bản thô.
.\.venv\Scripts\python.exe -X utf8 scripts/predict_current.py "ko được chửi người khác" "Cảm ơn bạn 👩🏽‍💻" "100k"
```

Nếu không có `--no-promote`, một lượt đầy đủ chỉ cập nhật mô hình hiện tại
khi kết quả validation tốt hơn baseline tham chiếu. `--quick` kiểm tra luồng
với 12 cấu hình và không được cập nhật mô hình hiện tại.

Lượt đã chạy: `reports/resplit_v1/steps23_v1/REPORT.md`.

- `validation_trials.csv`: mọi cấu hình, thời gian, lỗi và chỉ số.
- `validation_comparison.csv`: baseline tham chiếu và mô hình tốt nhất mỗi nhánh.
- `controlled_comparison.csv`: so sánh khi giữ cố định các tham số còn lại.
- `val_predictions_*.csv`: dự đoán validation theo ID và TP/FP/FN/TN.
- `training_summary.json`: hash nguồn/code, cấu hình, số đặc trưng, OOV và
  số vector rỗng; xác nhận test chưa được đánh giá trong lượt tối ưu.

Artifact tối ưu nhận danh sách văn bản thô và chứa cả tiền xử lý, bộ trích
đặc trưng và classifier. Khi chuyển sang máy khác cần kèm các module trong
`src/` và cài `requirements.txt`, `requirements-baselines.txt`.

Điểm validation dùng để chọn cấu hình, không phải điểm test. Kết quả test
của mô hình baseline trước đây vẫn thuộc lượt cũ và không được gán cho mô
hình tối ưu mới.

## Kiểm tra

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_preprocessing.py -v
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_text_features.py -v
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_optimize_text_features.py -v
.\.venv\Scripts\python.exe -X utf8 scripts/consolidate_data.py --check
```

Trường hợp biên: hai từ chỉ ngăn bởi tab/xuống dòng; emoji ghép và đơn vị
`100k` phải được giữ; chuỗi rỗng và từ chưa gặp phải có dự đoán hợp lệ.
