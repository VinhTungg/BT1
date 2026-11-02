# Thu thập bình luận PTIT để phân loại toxic / non-toxic

Bộ dữ liệu đang dùng: **resplit_v1**, được chọn trong [data/current_dataset.json](data/current_dataset.json).
Tối ưu bước 2–3: [quy trình và cách chạy](docs/TEXT_FEATURE_OPTIMIZATION.md).
Ba tập nằm ở `data/clean/`: train 6.000, validation 1.277, test 1.318 câu.
Xem [data/README.md](data/README.md) và [báo cáo tối ưu hiện tại](reports/resplit_v1/steps23_v2/REPORT.md).

Đợt thu thập thêm khoảng 6.000 bình luận: [docs/DATA_EXPANSION.md](docs/DATA_EXPANSION.md).
Chế độ tìm link và giới hạn số câu mới đều dùng chính `collect_comments.py`.

> **Trạng thái hiện tại:** Đã kiểm định ba cấu hình tham chiếu theo ba fold bài viết,
> thử tám biến thể tiền xử lý/đặc trưng và ngưỡng SVM học từ dự đoán ngoài mẫu trên train.
> Chọn chuẩn hóa viết tắt + TF-IDF từ 1–3/ký tự + từ dừng có chọn lọc + Linear SVM.
> Macro-F1 validation 68,86%; mô hình hiện tại nằm trong `models/current_run.json`.
> Mô hình tối ưu mới chưa đánh giá test. Bốn baseline + Dummy vẫn được giữ để đối chiếu.
> Toàn bộ 8.595 nhãn đã được nhóm chốt làm bộ nhãn chuẩn ngày 02/11/2026.
> Lịch sử và các đầu ra cũ được giữ cục bộ theo [archive/README.md](archive/README.md); Git chỉ chứa bộ dữ liệu và các lượt thực nghiệm còn dùng.

## ML baseline — Người 2

Đã có script huấn luyện bốn baseline và DummyClassifier trên dữ liệu đã chia.
Hướng dẫn cài đặt, tìm tham số, tải mô hình và bàn giao:
[docs/ML_BASELINES.md](docs/ML_BASELINES.md).

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pip install -r requirements-baselines.txt
.\.venv\Scripts\python.exe -X utf8 scripts/train_baselines.py
```

Kết quả validation hiện tại nằm trong `reports/resplit_v1/baselines/validation_report.md`,
các pipeline đã lưu trong `models/resplit_v1/baselines/`. Script chỉ fit trên train và
chọn tham số bằng validation; tập test dành cho đánh giá cuối.

Thử thêm TF-IDF ký tự, kết hợp từ + ký tự và trọng số lớp:

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/train_improved_baselines.py
```

Hướng dẫn và giới hạn kết luận:
[docs/ML_BASELINE_IMPROVEMENTS.md](docs/ML_BASELINE_IMPROVEMENTS.md).
Nhánh mở rộng khi chạy sẽ dùng cùng dataset đang chọn và xuất vào
`reports/resplit_v1/improvements/`, `models/resplit_v1/improvements/`.
Các thí nghiệm thuộc bộ chia cũ đã được dọn và giữ trong bản nén lịch sử cục bộ.

Python + Playwright. Bạn chọn bài bằng cách điền link vào `post_urls.txt` và tự đăng nhập Facebook.
Tool chỉ xử lý các bài trong danh sách này. Bộ nhận diện đã được chỉnh theo cấu trúc bình luận,
gắn thẻ và phản hồi nhiều tầng quan sát trên Facebook. Giao diện có thể thay đổi nên vẫn cần kiểm tra kết quả.

## 1. Cài đặt trên Windows

Mở terminal tại thư mục dự án, chạy:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m playwright install chromium
```

Không cần kích hoạt môi trường ảo bằng PowerShell.

## 2. Điền link bài đăng

Mở `post_urls.txt`, mỗi dòng một link thật. Dòng bắt đầu bằng `#` là ghi chú.

```text
https://www.facebook.com/groups/GROUP_ID/posts/POST_ID/
```

Thay `GROUP_ID`, `POST_ID` bằng giá trị thật, hoặc dán trực tiếp link đã sao chép từ Facebook.
Chấp nhận link nhóm dạng `/posts/`, `/permalink/`, link `permalink.php`, `story.php`, và `/share/p/`.
Link chia sẻ cần được trình duyệt chuyển hướng tới bài thật. Ưu tiên link trực tiếp `/groups/.../posts/.../`.
Không nhập link trang chủ nhóm. Bạn tự quyết định toàn bộ bài đăng nguồn.

## 3. Chạy tự động trên các bài bạn đã chọn

```powershell
.\.venv\Scripts\python.exe collect_comments.py --browser chrome --auto
```

1. Chrome riêng mở ra. Tự đăng nhập Facebook, xử lý xác minh nếu có.
2. Quay lại terminal, nhấn Enter để bắt đầu mở danh sách bài.
3. Tool lần lượt mở đúng các link trong file, chọn **Tất cả bình luận** nếu có, mở phản hồi và cuộn để tải thêm.
4. Mỗi đợt được lưu ngay vào `data/comments.csv`. Terminal in số văn bản nhận diện và số dòng mới đã lưu.
5. Mỗi bài dừng sau 3 lượt không có thêm nội dung/mục bình luận, hoặc khi đạt giới hạn 120 lượt.
   Nếu chạm giới hạn, tool báo rõ có thể còn bình luận chưa tải. Dùng Ctrl+C để dừng sớm, các đợt đã lưu vẫn còn.

Tăng giới hạn cho bài có nhiều bình luận:

```powershell
.\.venv\Scripts\python.exe collect_comments.py --browser chrome --auto --max-rounds 300
```

Nếu mạng tải chậm, có thể thêm `--wait-ms 2000 --idle-rounds 5`.
Dùng `--browser msedge` cho Edge, hoặc bỏ `--browser chrome` để dùng Chromium đã cài bằng Playwright.
Tool không tự tìm bài khác và dừng lấy nếu trang chuyển sang một bài không khớp nguồn đã chọn.

### Chế độ xem trước rồi xác nhận lưu

```powershell
.\.venv\Scripts\python.exe collect_comments.py --browser chrome
```

Ở mỗi bài, quay về terminal:
   - **Enter**: đọc các bình luận hiện đang được tải và xem trước 10 dòng.
   - **s** ở bước xem trước: xác nhận lưu các dòng mới.
   - **a**: chọn bộ lọc, tự mở thêm/cuộn theo giới hạn trên, tích lũy rồi xem trước để lưu.
   - **n**: chuyển sang bài tiếp theo.
   - **q** hoặc Ctrl+C: dừng.

Bạn cũng có thể tự mở bình luận trên trình duyệt, rồi nhấn Enter và lưu nhiều lần trong cùng bài.

Nếu Facebook gỡ các bình luận cũ khỏi DOM khi cuộn, chương trình chỉ đọc được phần còn hiện diện.
Chế độ `--auto` lưu từng đợt để giảm mất dữ liệu. Nếu giao diện không nhận diện được,
dùng chế độ xem trước và thao tác trực tiếp trên trình duyệt.
Không bấm Thích, gửi bình luận hoặc trả lời tự động.

## 4. Kết quả và chạy tiếp

CSV mặc định: `data/comments.csv`, mã hóa UTF-8 có BOM để đọc được tiếng Việt.

| Cột | Nội dung |
|---|---|
| id | Mã băm nội dung do chương trình tạo, không phải ID người dùng Facebook |
| post_id | Mã nguồn bài đăng; link chia sẻ chưa được giải quyết dùng mã băm URL |
| text | Nội dung sau chuẩn hóa Unicode và khoảng trắng |
| label | Để trống; bạn gán `0` = non-toxic, `1` = toxic sau |
| source_url | Link bài đăng đã bỏ tham số theo dõi |
| collected_at | Thời điểm thu thập, UTC |
| extraction_method | Cách nhận diện bình luận, để hỗ trợ kiểm tra |

Thẻ tên trong nội dung được thay bằng `[MENTION]`, liên kết khác bằng `[URL]`, giữ phần câu còn lại.
Không chủ động trích tên/ID/URL hồ sơ người bình luận. Nội dung bình luận có thể tự chứa tên,
số điện thoại hoặc thông tin khác; cần kiểm tra, ẩn thông tin cá nhân trong bản dữ liệu để sử dụng/chia sẻ.

Chạy lại sẽ đọc CSV cũ và **thêm dòng mới**, giữ nguyên nhãn đã gán.
Nội dung giống hệt sau chuẩn hóa được loại trùng trên toàn bộ CSV, kể cả ở các bài khác nhau.
Vì vậy số dòng là số văn bản khác nhau, không phải tổng số lượt bình luận trên Facebook.
Không gán nhãn dựa trên từ tục tự động. Xem lại dữ liệu trước khi huấn luyện, nhất là các dòng
`reply_action_heuristic`, bình luận có link/mention và các phản hồi cần ngữ cảnh.

Để lưu một bộ dữ liệu khác:

```powershell
.\.venv\Scripts\python.exe collect_comments.py --output data/dot_2.csv
```

## 5. Khi lấy được 0 dòng hoặc nội dung không đúng

### Trang trắng ở bước xác minh Meta / Arkose

Nếu địa chỉ có `/two_step_verification/` hoặc `/checkpoint/`, Facebook đang yêu cầu xác minh;
chương trình chưa đến bước thu thập. Ảnh trang trắng không đủ để kết luận do mạng, trình duyệt hay Facebook.

1. Thử tải lại trang một lần và tự hoàn tất kiểm tra khi nó hiện ra.
2. Nếu vẫn trắng, dừng chương trình bằng Ctrl+C ở terminal rồi thử Chrome đã cài trên máy:

   ```powershell
   .\.venv\Scripts\python.exe collect_comments.py --browser chrome --login-only
   ```

   Hoặc dùng Edge: thay `chrome` bằng `msedge`. Đây là lựa chọn trình duyệt để kiểm tra
   tương thích, không bảo đảm khắc phục trang xác minh trắng.
3. Chương trình dùng profile riêng cho từng trình duyệt. Tự đăng nhập và hoàn tất xác minh;
   chỉ nhấn Enter trong terminal khi đã thấy Bảng tin/bài đăng. Nếu còn ở trang xác minh,
   chương trình sẽ chờ tiếp, không điều hướng sang bài đăng. Có thể gõ `q` để thoát và giữ phiên.
4. Sau khi đăng nhập xong, chạy thu thập bằng **cùng trình duyệt/profile**:

   ```powershell
   .\.venv\Scripts\python.exe collect_comments.py --browser chrome
   ```

`--login-only` không yêu cầu file link đã điền, không đọc/ghi CSV.
Nếu dùng `--profile` tùy chỉnh thì truyền lại cùng đường dẫn ở lần sau.
Không xóa profile cũ để thử lỗi; không dùng thư mục hồ sơ Chrome hằng ngày đang mở.
Nếu Chrome/Edge thông thường cũng không hiển thị được bước xác minh, dùng các lựa chọn
trợ giúp/đăng nhập chính thức của Facebook trước khi tiếp tục. Không có chức năng vượt xác minh.

### Không nhận diện được bình luận

- Kiểm tra đã đăng nhập, mở đúng bài, có quyền xem, và đã tải bình luận.
- Không lưu nếu bản xem trước chứa nội dung bài đăng, tên người hoặc nút giao diện.
- Facebook có thể thay đổi DOM. Kiểm tra vùng bài và phần văn bản bằng DevTools trước khi đổi selector.
- Có thể truyền `--scope-selector`, `--article-selector`, `--text-selector`.
  `scope` phải khớp đúng một vùng hiển thị; `article` phải là từng bình luận, `text` là phần nội dung.
  Khi dùng selector tùy chỉnh, kiểm tra kiểu bình luận bằng nhãn/nút Reply sẽ được bỏ qua,
  nên cần kiểm tra bản xem trước kỹ hơn.
- Mặc định bỏ qua ảnh/sticker không có văn bản và câu chỉ gắn thẻ/link sau khi ẩn thông tin.
  Các bình luận dài chưa bung hết hoặc dạng DOM mới có thể bị bỏ sót.
  Không suy ra nhãn non-toxic cho các bình luận không lấy được.
- Không bảo đảm lấy toàn bộ bình luận; ghi rõ số bài, số dòng thực lấy và giới hạn này trong báo cáo.
- Số Facebook hiển thị có thể bao gồm phản hồi, ảnh, sticker, nội dung không còn truy cập được và văn bản trùng,
  nên không nhất thiết bằng số dòng CSV.
- Mặc định chỉ đọc vùng hộp thoại bài đăng, tránh lấy nhầm bình luận ở bảng tin phía sau.
  Nếu giao diện dùng trang bài viết riêng không có hộp thoại, xác định vùng đúng rồi truyền `--scope-selector`.
- Sau khi sửa code, dừng tiến trình cũ bằng Ctrl+C và chạy lại để nhận thay đổi.

## 6. Phiên đăng nhập và nguồn dữ liệu

Chỉ dùng với nguồn bạn có quyền thu thập/sử dụng và phù hợp điều kiện của Facebook.
Quyền xem một nhóm không tự động tương đương quyền thu thập tự động.
Không có cơ chế vượt CAPTCHA, checkpoint, quyền truy cập hoặc giới hạn của Facebook.
Gặp xác minh thì tự xử lý trong trình duyệt; nếu bị chặn thì dừng.

Phiên Chromium nằm trong `.browser-profile/`; Chrome dùng `.browser-profile-chrome/`,
Edge dùng `.browser-profile-msedge/`. Giữ riêng các thư mục này, không chia sẻ/commit.
Không dùng chung profile với Chrome đang mở. CSV và profile đã được thêm vào `.gitignore`.

## Kiểm tra

Kiểm tra xử lý link, lưu/loại trùng CSV, giữ nhãn, lọc đúng bài, mở phản hồi tải chậm,
giới hạn lượt và bộ trích xuất trên trang HTML giả lập:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Kiểm tra HTML cần Playwright và Chromium; nếu chưa cài thì test đó sẽ được bỏ qua.
Các ví dụ trong test là dữ liệu tổng hợp, không được đưa vào bộ dữ liệu thật.

Tài liệu: [Playwright Python](https://playwright.dev/python/docs/library),
[Browser contexts](https://playwright.dev/python/docs/api/class-browsertype#browser-type-launch-persistent-context),
[Meta về scraping](https://www.facebook.com/help/463983701520800).
