# Bốn baseline bắt buộc — Người 2

Bộ dữ liệu `resplit_v1`: train 6.000, validation 1.277, test 1.318 câu.
BoW/TF-IDF kết hợp Multinomial Naive Bayes/Linear SVM tạo bốn baseline;
DummyClassifier luôn đoán lớp phổ biến dùng làm mốc đối chiếu.

## Chạy

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pip install -r requirements-baselines.txt
# Ghi lượt mới để giữ kết quả đã chốt.
.\.venv\Scripts\python.exe -X utf8 scripts/train_baselines.py --model-dir models/resplit_v1/baselines_repeat --report-dir reports/resplit_v1/baselines_repeat
```

Mặc định script đọc `data/current_dataset.json`. Đầu vào là
`text_tokenized`; chỉ fit trên train, dùng validation chọn tham số theo
Macro-F1, rồi F1-toxic/Recall-toxic khi hòa điểm. Không đọc test và không
refit train+validation trong bước chọn mô hình.

Bộ trích đặc trưng dùng `str.split` trên văn bản đã tách từ, giữ từ ghép có
underscore, emoji và token một ký tự. Chữ thường được áp dụng trong vectorizer.

## Sản phẩm giữ lại

- `models/resplit_v1/baselines/`: bốn pipeline, Dummy và manifest.
- `reports/resplit_v1/baselines/`: kết quả, cấu hình đã thử và dự đoán validation.
- `reports/resplit_v1/final_evaluation/`: đánh giá test của lượt baseline cũ.

Manifest ghi hash đầu vào, artifact, phiên bản thư viện và kiểm tra nạp lại.
Dự đoán validation giữ ID để phân tích lỗi. Test chỉ dùng sau khi chốt cấu hình.

```python
import joblib
pipeline = joblib.load('models/resplit_v1/baselines/tfidf_svm.joblib')
print(pipeline.predict(['Sinh_viên đang đóng học_phí']))
```

Mô hình đang dùng đã được tối ưu thêm ở bước 2–3. Với câu mới ở dạng thô,
dùng `scripts/predict_current.py` và đọc `models/current_run.json`;
không gán điểm test của baseline cũ cho mô hình hiện tại.
Xem [hướng dẫn tối ưu](TEXT_FEATURE_OPTIMIZATION.md).
