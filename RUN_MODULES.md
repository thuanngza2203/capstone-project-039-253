# Hướng dẫn chạy nhanh các module

Luồng ứng dụng: **Web → Detection → RAG → vLLM trên Vast.ai**.

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

Chuyển cổng `8001` trên Windows tới vLLM cổng `8000` trên Vast. Lệnh dưới dùng instance hiện tại (`92.180.27.84`, SSH port `59921`) và key `vast_ed25519`; cập nhật IP/port theo Connect của Vast khi đổi instance.

```powershell
ssh -i "$env:USERPROFILE\.ssh\vast_ed25519" -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -p 59921 -L 127.0.0.1:8001:127.0.0.1:8000 root@92.180.27.84
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
