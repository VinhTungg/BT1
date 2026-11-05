# PhoBERT Best Model Checkpoint — models/phobert_best/

* **Mô hình kiến trúc:** `vinai/phobert-base-v2` (`RobertaForSequenceClassification`)
* **Số lớp phân loại:** 2 (`0`: Non-toxic, `1`: Toxic)
* **Kích thước từ vựng (Vocab size):** 64.003
* **Special Tokens bổ sung:**
  * `[MENTION]` $\rightarrow$ Token ID: `64001`
  * `[URL]` $\rightarrow$ Token ID: `64002`
* **Trọng số nhị phân:** `model.safetensors` (540,029,528 bytes)
* **SHA-256 Checkpoint:** `4f56699fc2bc39a87a446cddcaaf1e86650f180c5cc506729a2a6b6469c5e740`
* **Ngưỡng quyết định tối ưu:** $\tau = 0.52$

---

## 1. Cấu trúc thư mục

Thư mục này lưu trữ checkpoint đầy đủ, cấu hình kiến trúc, bảng mã và tokenizer đã được đồng bộ hóa hoàn chỉnh:
* `model.safetensors`: File trọng số fine-tuned hoàn chỉnh (540 MB), đã được nạp và kiểm thử tải thành công.
* `config.json`: Cấu hình kiến trúc PhoBERT base v2, `num_labels = 2`, `vocab_size = 64003`.
* `tokenizer_config.json`: Cấu hình tokenizer PhoBERT kèm ánh xạ `added_tokens_decoder` cho `[MENTION]` và `[URL]`.
* `added_tokens.json`: Ánh xạ ID `<mask\>` (64000), `[MENTION]` (64001), `[URL]` (64002).
* `vocab.txt` & `bpe.codes`: Bộ từ vựng BPE tiếng Việt gốc của PhoBERT.

---

## 2. Lưu ý về Quản Lý Mã Nguồn Git

Theo quy định quản lý mã nguồn của dự án (khai báo trong [.gitignore](../../.gitignore)):
* File nhị phân lớn `model.safetensors` (~540MB) được `.gitignore` loại trừ để tránh phình to Git tree lịch sử.
* File zip tổng hợp `phobert_deliverables_nguoi_3.zip` được lưu trữ tại máy hoặc Google Drive để bàn giao trực tiếp cho Người 4.
* Toàn bộ cấu hình tokenizer, vocab, json metadata và báo cáo phân tích đều được commit và kiểm soát phiên bản trong Git.

---

## 3. Hướng Dẫn Nạp Checkpoint & Inference

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_dir = "models/phobert_best"
threshold = 0.52

# 1. Nạp tokenizer và model
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
model.eval()

# 2. Tokenize câu đầu vào đã tiền xử lý
text = "[MENTION] bài viết này rất bổ ích, cảm ơn bạn!"
inputs = tokenizer(text, return_tensors="pt", max_length=128, padding=True, truncation=True)

# 3. Suy luận xác suất với ngưỡng 0.52
with torch.no_grad():
    outputs = model(**inputs)
    prob_toxic = torch.softmax(outputs.logits, dim=1)[:, 1].item()
    prediction = int(prob_toxic >= threshold)

print(f"Xác suất toxic: {prob_toxic:.4f} | Dự đoán: {'Toxic' if prediction == 1 else 'Non-toxic'}")
```
