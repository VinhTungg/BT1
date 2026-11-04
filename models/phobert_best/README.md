# PhoBERT Best Model Checkpoint — models/phobert_best/

* **Mô hình kiến trúc:** `vinai/phobert-base-v2` (`RobertaForSequenceClassification`)
* **Số lớp phân loại:** 2 (`0`: Non-toxic, `1`: Toxic)
* **Kích thước từ vựng (Vocab size):** 64.003
* **Special Tokens bổ sung:**
  * `[MENTION]` $\rightarrow$ Token ID: `64001`
  * `[URL]` $\rightarrow$ Token ID: `64002`

---

## 1. Cấu trúc thư mục

Thư mục này lưu trữ cấu hình kiến trúc, bảng mã và tokenizer đã được đồng bộ hóa hoàn chỉnh:
* `config.json`: Cấu hình kiến trúc PhoBERT base v2, `num_labels = 2`, `vocab_size = 64003`.
* `tokenizer_config.json`: Cấu hình tokenizer PhoBERT kèm ánh xạ `added_tokens_decoder` cho `[MENTION]` và `[URL]`.
* `added_tokens.json`: Ánh xạ ID `<mask\>` (64000), `[MENTION]` (64001), `[URL]` (64002).
* `vocab.txt` & `bpe.codes`: Bộ từ vựng BPE tiếng Việt gốc của PhoBERT.
* `special_tokens_map.json`: Khai báo danh sách các special tokens bổ sung.

---

## 2. Lưu ý về file trọng số lớn (`model.safetensors` / `pytorch_model.bin`)

Theo quy định quản lý mã nguồn của dự án (khai báo trong [.gitignore](../../.gitignore)):
* Các file nhị phân lớn (`*.safetensors`, `*.bin`, `*.zip`) không được commit trực tiếp vào Git tree nhằm tránh phình to dung lượng repository.
* Trọng số fine-tune đầy đủ (`model.safetensors` ~540MB) được sinh ra từ quá trình huấn luyện trên Colab GPU T4.
* Khi tải lại checkpoint trên môi trường mới:
  * File weights có thể nạp trực tiếp qua Google Drive hoặc link checkpoint đã công bố trong báo cáo BTL.
  * SHA-256 Checkpoint: Được ghi nhận và đối chiếu tương ứng trong `reports/phobert/run_manifest.json`.

---

## 3. Hướng dẫn nạp Tokenizer & Inference

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_dir = "models/phobert_best"

# 1. Nạp tokenizer đã đăng ký special token
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
assert len(tokenizer) == 64003

# 2. Tokenize câu có chứa tag [MENTION]
text = "[MENTION] Chúc bạn một ngày tốt lành"
inputs = tokenizer(text, return_tensors="pt")
# [MENTION] được ánh xạ chuẩn xác thành token ID 64001, không bị cắt nhỏ
```

