# PlantGPT — nhận diện và tư vấn bệnh cây

Luồng ứng dụng: **Web → Detection → RAG → LLM**. LLM có thể dùng Ollama,
Gemini hoặc vLLM trên server Linux/Vast.ai.

## Sau khi clone lần đầu

Repo có source, tài liệu `.txt` để index, `.env.example`, requirements và
`package-lock.json`. Cần cài **Python 3.11**, **Node.js 22** và có **MongoDB**
đang chạy. Mỗi module Python dùng một `.venv` riêng.

Không đưa lên Git: `.env`, trọng số model, `.venv`, `node_modules`, Chroma index
và dữ liệu hội thoại MongoDB. Vì vậy clone xong cần cài dependencies, điền cấu
hình, tải model và tạo index trước khi chat được.

Các lệnh PowerShell dưới đây bắt đầu từ **thư mục gốc repo**.

### 1. RAG

```powershell
cd RAG-module
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Sửa `.env`: chọn `LLM_PROVIDER`, cấu hình provider đó; đặt `EMBEDDING_DEVICE=cpu`
nếu máy không có CUDA. Nếu dùng Ollama, cần chạy Ollama và tải `OLLAMA_MODEL`
trước khi hỏi. Nếu dùng vLLM, cần server đang chạy, URL truy cập được, API key
và tên model khớp server. Hướng dẫn: [RAG](RAG-module/README.md),
[vLLM/Vast.ai](LLM-server-module/README.md).

Tạo index theo `CHUNKING_STRATEGY` trong `.env`, rồi bật API:

```powershell
.\.venv\Scripts\python.exe main.py index
.\.venv\Scripts\python.exe -m server
```

Lần index đầu cần Internet để tải embedding/tokenizer. Đổi provider LLM không
cần index lại. API: `http://127.0.0.1:8010/docs`.

### 2. Detection — terminal khác tại thư mục gốc repo

```powershell
cd detection-server-module
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Điền `GROQ_API_KEY`, `MONGO_URI`, `ADMIN_PASSWORD`; cấu hình để nối với RAG:

```dotenv
ANSWER_BACKEND=rag
RAG_API_URL=http://127.0.0.1:8010
# Để trống: dùng LLM_PROVIDER bên RAG. Chỉ liệt kê provider đã cấu hình và chạy được.
CHAT_LLM_PROVIDERS=
```

Nếu đặt `RAG_API_KEY`, hai module phải dùng cùng key. Khởi động MongoDB trước,
sau đó chạy:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8005
```

Detection tải checkpoint còn thiếu vào `detection-server-module/models/`.
`app/plant_ai/models/` chứa **code kiến trúc**, phải được commit cùng source.
Với `ANSWER_BACKEND=rag`, không cần index thư mục `rag/` nội bộ của Detection.
Hướng dẫn khác, gồm backend Groq độc lập: [Detection README](detection-server-module/README.md).

### 3. Web — terminal khác tại thư mục gốc repo

```powershell
cd application/web
npm.cmd ci
npm.cmd run dev
```

Mở `http://localhost:5173`. Cấu hình mặc định đã proxy tới Detection `8005` và
RAG `8010`; chỉ cần `.env.local` nếu đổi địa chỉ backend. Xem
[Web README](application/web/README.md) khi build hoặc deploy.

## Những lần chạy sau

Xem [RUN_MODULES.md](RUN_MODULES.md) để bật từng server. Không cần cài lại
dependencies hoặc index lại khi source, dữ liệu và cấu hình index không đổi.

## File cần giữ trong Git

- Source của các module, gồm `detection-server-module/app/plant_ai/models/`
  và `application/web/src/lib/`.
- `RAG-module/data/`, `detection-server-module/rag/data/`, static assets.
- `.env.example`, requirements, `package.json`, `package-lock.json`, script
  cài/chạy, scripts đánh giá và tài liệu hướng dẫn.
- `agents/`, báo cáo sinh tự động, cache và kết quả thí nghiệm không bắt buộc
  để chạy ứng dụng. OpenAPI hiện tại có tại `http://127.0.0.1:8010/openapi.json`.

`.gitignore` chỉ quyết định file chưa track nào sẽ bị bỏ qua; sửa rule không
tự thêm file vào commit hoặc xóa file đã commit. Trước khi push, kiểm tra
`git status --short` và thêm cả các file source mới hiện ra.
