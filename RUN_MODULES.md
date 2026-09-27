# Hướng dẫn chạy nhanh các module

Luồng ứng dụng: **Web → Detection → RAG → vLLM trên Vast.ai**.

## Cách nhanh: chạy bằng Docker

Một lệnh bật Web, Detection, RAG và SSH tunnel tới Vast (4 container, file [docker-compose.yml](docker-compose.yml)). vLLM vẫn chạy trên Vast (mục 1 bên dưới); MongoDB vẫn là service đang chạy trên Windows, dữ liệu cũ giữ nguyên.

**Cần có:** Docker Desktop (backend WSL2), MongoDB chạy trên Windows; instance Vast đang chạy vLLM nếu dùng Qwen 27B. Máy không có GPU NVIDIA: bỏ `#` ở dòng `COMPOSE_FILE` trong `.env` gốc.

**Lần đầu:** chép `.env.example` ở thư mục gốc thành `.env` (hoặc để `run.cmd` tự chép), điền `VAST_SSH_KEY_FILE` nếu dùng Vast; `HF_CACHE_DIR` để dùng lại model HuggingFace đã tải trên máy (tùy chọn). `.env` của từng module vẫn dùng như cũ; Docker chỉ đổi các địa chỉ nội bộ (RAG gọi `vast-tunnel:8000`, Detection gọi `rag:8010` và MongoDB qua `host.docker.internal`).

**Bật:** nhấp đúp `run.cmd`. Script hỏi lệnh SSH của Vast:

```text
Vast dang luu: root@92.180.27.84, cong SSH 59921
Dan lenh SSH cua Vast (nut Connect), Enter de giu nguyen: ssh -p 41234 root@173.44.12.9 -L 8080:localhost:8080
```

Mỗi lần thuê GPU mới, Vast cấp IP và cổng SSH mới: bấm **Connect** trên Vast, chép dòng `ssh -p ... root@...` rồi dán vào (dán nguyên dòng, script tự tách IP và cổng, lưu vào `.env` gốc). Vẫn instance cũ thì chỉ cần Enter. Đổi IP khi hệ thống đang chạy cũng chỉ cần chạy lại `run.cmd`: Docker chỉ tạo lại container `vast-tunnel`.

**Tắt:** nhấp đúp `stop.cmd`, hoặc `docker compose down`.

**Tự build và chạy bằng lệnh** (thấy được tiến độ tải; lần đầu mất khá lâu vì phải tải vài GB thư viện, các lần sau dùng lại):

```powershell
cd D:\HCMUT\Capstone_Project
docker compose build detection   # từng image, hoặc "docker compose build" cho cả 4
docker compose build web
docker compose up -d             # bật cả hệ thống
docker compose ps                # cột STATUS: running / healthy
```

Build bị ngắt giữa chừng (mất mạng, tắt máy) thì chạy lại đúng lệnh đó: các gói đã tải được giữ trong cache của Docker, không tải lại từ đầu.

Web ở `http://localhost:5173` (điện thoại cùng Wi-Fi: `http://<IP máy>:5173`), Swagger ở `http://127.0.0.1:8005/docs` và `http://127.0.0.1:8010/docs`.

| Việc | Lệnh |
|---|---|
| Xem log | `docker compose logs -f rag` (hoặc `detection`, `web`, `vast-tunnel`) |
| Trạng thái | `docker compose ps` |
| Đổi instance Vast | chạy lại `run.cmd`, dán lệnh SSH mới (hoặc sửa `VAST_SSH_HOST`, `VAST_SSH_PORT` trong `.env` gốc rồi `docker compose up -d`) |
| Kiểm tra tunnel tới Vast | `docker compose logs vast-tunnel` |
| Index lại sau khi đổi dữ liệu/chunking/embedding | `docker compose stop rag`, rồi `docker compose run --rm rag python main.py index`, rồi `docker compose start rag` |
| Sửa `.env` của một module | `docker compose restart rag` (hoặc `detection`) |

Lần bật đầu, container RAG tự tạo index (khoảng một phút) trong volume Docker; các lần sau dùng lại. Model nhận diện nằm ở `detection-server-module/models`, model HuggingFace ở `HF_CACHE_DIR` (hoặc volume `hf-cache`): tải một lần, các lần sau dùng lại.

## Cách chạy tay từng module

Mỗi server chạy trong một terminal riêng và giữ terminal đó mở. Các lệnh dưới dành cho máy đã cài dependencies, tạo `.venv` riêng cho từng module Python và cấu hình `.env`. MongoDB phải đang chạy, `MONGO_URI` trỏ đúng địa chỉ/cổng.

## 1. LLM-server-module — chạy trên Vast.ai (Linux)

Phục vụ model qua API vLLM. Nếu model đã chạy thì giữ tiến trình hiện có.

```bash
cd /workspace/capstone-project-039-253/LLM-server-module
source .venv/bin/activate
python serve.py
```

Kiểm tra bằng terminal Linux khác:

```bash
cd /workspace/capstone-project-039-253/LLM-server-module
source .venv/bin/activate
python check_api.py
```

## 2. SSH tunnel — terminal riêng trên Windows

Chuyển cổng `8001` trên Windows tới vLLM cổng `8000` trên Vast. Mỗi lần thuê GPU, Vast cấp IP và cổng SSH mới: thay `<IP>`, `<PORT>` theo lệnh ở nút Connect của Vast.

```powershell
ssh -i "$env:USERPROFILE\.ssh\vast_ed25519" -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -p <PORT> -L 127.0.0.1:8001:127.0.0.1:8000 root@<IP>
```

Với tunnel này, sửa các dòng tương ứng trong `RAG-module/.env`:

```dotenv
LLM_PROVIDER=vllm
VLLM_BASE_URL=http://127.0.0.1:8001/v1
VLLM_HOST=
VLLM_PORT=
VLLM_MODEL=rag-llm
```

`VLLM_MODEL` phải khớp `LLM_SERVED_MODEL_NAME`; `VLLM_API_KEY` phải khớp `LLM_API_KEY` trên Vast. Nếu dùng API public, đặt URL đó vào `VLLM_BASE_URL` và bỏ qua tunnel.

## 3. RAG-module — terminal Windows

Truy xuất tài liệu và gọi LLM để sinh câu trả lời. API ở `http://127.0.0.1:8010/docs`.

```powershell
cd D:\HCMUT\Capstone_Project\RAG-module
.\.venv\Scripts\Activate.ps1
python -m server
```

Chỉ khi chưa có index hoặc đã đổi dữ liệu/chunking/embedding, chạy lệnh này **trước khi bật RAG API**. Strategy lấy từ `.env`:

```powershell
python main.py index
```

## 4. detection-server-module — terminal Windows khác

**Cài lần đầu nếu chưa có `.venv`:**

```powershell
cd D:\HCMUT\Capstone_Project\detection-server-module
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Lệnh activate chỉ bật môi trường đã có, không tạo `.venv`. Đường dẫn đúng là `.\.venv\Scripts\Activate.ps1`. Những lần chạy sau bỏ qua bước tạo venv và cài dependencies.

Nhận câu hỏi/ảnh, chuẩn hóa query, gọi RAG và lưu hội thoại vào MongoDB. Trong `.env`, đặt:

```dotenv
ANSWER_BACKEND=rag
RAG_API_URL=http://127.0.0.1:8010
```

Điền `GROQ_API_KEY` cho bước chuẩn hóa câu hỏi. Nếu bật `RAG_API_KEY`, đặt cùng key ở cả detection và RAG.

```powershell
cd D:\HCMUT\Capstone_Project\detection-server-module
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.api:app --host 127.0.0.1 --port 8005
```

## 5. application/web — terminal Windows khác

Giao diện chat và quản trị. Mở `http://localhost:5173` sau khi chạy:

```powershell
cd D:\HCMUT\Capstone_Project\application\web
npm.cmd run dev
```

## 6. Kiểm tra riêng RAG — không cần mở Web

Chạy từng lệnh cần dùng trong terminal khác; `search` kiểm tra retrieval, `ask` hỏi một câu, `chat` hỏi liên tục.

```powershell
cd D:\HCMUT\Capstone_Project\RAG-module
.\.venv\Scripts\Activate.ps1
python ..\LLM-server-module\check_api.py --env-file .env
python main.py search "Bệnh ghẻ táo xử lý như thế nào?" --debug
python main.py ask "Bệnh ghẻ táo xử lý như thế nào?"
python main.py chat --debug
```

Cài lần đầu: [LLM-server](LLM-server-module/README.md) · [RAG](RAG-module/README.md) · [Detection](detection-server-module/README.md) · [Web](application/web/README.md).
