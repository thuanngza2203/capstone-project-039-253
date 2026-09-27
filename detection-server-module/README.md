# Detection server

API chat của hệ thống (FastAPI, cổng 8005). Nhận câu hỏi và ảnh lá cây từ web, rồi:

1. Nhận diện ảnh: ConvNeXt-Tiny đoán loại cây → GenYOLO tách lá → IEViT đoán bệnh.
2. Chuẩn hóa câu hỏi bằng Groq (sửa dấu, chính tả, viết tắt; trích tên cây, bệnh).
3. Quyết định trả lời, hỏi lại hay xin ảnh; nếu trả lời thì gọi [RAG module](../RAG-module/README.md).
4. Lưu hội thoại, đánh giá thích/không thích và dữ liệu quản trị vào MongoDB.

Nhận diện bệnh từ ảnh hỗ trợ 9 loại cây: táo, anh đào, ngô, nho, đào, ớt chuông, khoai tây, dâu tây, cà chua. Ảnh không có lá cây bị từ chối.

## Cài đặt lần đầu

Cần Python 3.11, MongoDB đang chạy và [Groq API key](https://console.groq.com). Lần chạy đầu cần Internet để tải model nhận diện (khoảng 740 MB) vào `models/`.

```powershell
cd detection-server-module
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# PyTorch bản CPU (nhẹ, đủ dùng). Có GPU thì chọn bản CUDA ở pytorch.org.
python -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

## Cấu hình `.env`

| Biến | Bắt buộc | Ý nghĩa |
|---|---|---|
| `GROQ_API_KEY` | Có | Key Groq cho bước chuẩn hóa câu hỏi |
| `MONGO_URI` | Có | Mặc định `mongodb://localhost:27017` |
| `ADMIN_PASSWORD` | Có, nếu dùng trang quản trị | Tài khoản `ADMIN_USERNAME` (mặc định `admin`) |
| `ANSWER_BACKEND` | | `rag` (mặc định): gọi RAG module. `groq`: dùng `rag/` nội bộ + Groq, xem cuối trang |
| `RAG_API_URL` | | Mặc định `http://127.0.0.1:8010` |
| `RAG_API_KEY` | | Chỉ khi RAG module đặt `RAG_API_KEY`; hai bên phải trùng nhau |
| `CHAT_LLM_PROVIDERS` | | Model người dùng chọn trên web, ví dụ `vllm,gemini`. Chỉ liệt kê model đã cấu hình bên RAG; để trống thì dùng `LLM_PROVIDER` của RAG |
| `WEB_SEARCH_MODEL` | | Model Groq cho nút "Tìm trên web"; để trống thì ẩn nút |

## Chạy

```powershell
uvicorn app.api:app --host 127.0.0.1 --port 8005
```

`python main.py` cũng chạy được, có tự nạp lại khi sửa code (dành cho lúc phát triển).

| Địa chỉ | Nội dung |
|---|---|
| <http://127.0.0.1:8005/docs> | Swagger: thử mọi API |
| <http://127.0.0.1:8005/health> | Trả `{"status":"ok"}` khi server chạy |
| <http://127.0.0.1:8005> | Trang chat đơn giản có sẵn; giao diện chính là [web](../application/web/README.md) |

Thử gửi câu hỏi kèm ảnh (trường `image` không bắt buộc):

```powershell
curl.exe -X POST http://127.0.0.1:8005/api/chat `
  -F "session_id=demo-01" `
  -F "message=Lá cà chua này bị bệnh gì?" `
  -F "image=@C:\duong-dan\la-ca-chua.jpg"
```

## API chính

| Method | Endpoint | Việc |
|---|---|---|
| `POST` | `/api/chat` | Gửi câu hỏi và ảnh (`multipart/form-data`) |
| `GET` | `/api/models` | Các model người dùng chọn được, có nút "Tìm trên web" không |
| `GET`, `DELETE` | `/api/conversations[/{session_id}]` | Danh sách, chi tiết, xóa cuộc trò chuyện |
| `PUT` | `/feedback` | Đánh giá thích / không thích một câu trả lời |
| `POST` | `/admin/login` | Đăng nhập quản trị, trả token |
| `GET`, `POST`, `DELETE` | `/admin/reviews`, `/admin/training`, `/admin/review/{id}` | Duyệt đánh giá, đưa câu trả lời mẫu vào Feedback RAG |

Các route `/admin/*` (trừ `/admin/login`) cần đăng nhập khi `ADMIN_PASSWORD` có giá trị.

## Kiểm thử

```powershell
python -m pytest -q tests
```

Test chạy offline: không cần Groq, MongoDB, RAG hay model nhận diện.

## Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| Mọi lượt chat báo lỗi Groq 400 | `NORMALIZER_REASONING_EFFORT` không hợp model: `openai/gpt-oss-*` nhận `low`, `medium`, `high` |
| Chat trả 503 | RAG module chưa chạy, sai `RAG_API_URL`, hoặc sai `RAG_API_KEY` |
| Chat hoặc trang lịch sử báo lỗi | MongoDB chưa chạy hoặc sai `MONGO_URI` |
| Ảnh bị báo "Ảnh không hợp lệ" | Ảnh không thấy lá; chụp gần lá hơn, hoặc giảm `LEAF_MIN_CONFIDENCE` |
| Lượt có ảnh đầu tiên rất chậm | Đang tải/nạp model nhận diện, các lượt sau nhanh hơn |

## Cấu trúc

```text
app/
  api.py          Tạo FastAPI app và các route chat
  chat/           Chuẩn hóa câu hỏi, điều hướng, ghép ngữ cảnh, pipeline
  answer/         Gọi RAG module (rag_http.py) hoặc Groq
  plant_ai/       Nhận diện cây, tách lá, nhận diện bệnh (models/ ở đây là code kiến trúc)
  feedback/       MongoDB: hội thoại, đánh giá, Feedback RAG
  prompts/        Prompt chuẩn hóa và sinh câu trả lời
  routes/         Route quản trị, đăng nhập, feedback
  static/         Trang chat và quản trị đơn giản
rag/              RAG nội bộ, chỉ dùng khi ANSWER_BACKEND=groq
models/           Checkpoint tự tải ở lần chạy đầu (không có trong Git)
tests/            Test offline
```

**Chế độ `ANSWER_BACKEND=groq`** (không cần RAG module): tạo index nội bộ một lần bằng `python -m rag.ingest --reset`, câu trả lời do Groq sinh từ `rag/data/`.
