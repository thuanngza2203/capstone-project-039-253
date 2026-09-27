# PlantGPT — Chatbot nhận diện và tư vấn bệnh cây trồng

Người dùng hỏi bằng tiếng Việt hoặc gửi ảnh lá cây. Hệ thống nhận diện cây và bệnh từ ảnh, tìm thông tin trong kho 25 tài liệu bệnh cây (10 loại cây), rồi sinh câu trả lời có dẫn nguồn.

## Kiến trúc

```text
Trình duyệt ─► Web ─────────► Detection server ─────► RAG module ─────► LLM server
               (React)        nhận diện ảnh lá,        tìm tài liệu,      vLLM trên GPU Vast.ai
               :5173          chuẩn hóa câu hỏi        sinh câu trả lời   (hoặc Gemini / Ollama)
                              (Groq), MongoDB          :8010
                              :8005
```

| Thư mục | Vai trò | Hướng dẫn |
|---|---|---|
| [`application/web`](application/web) | Giao diện chat và trang quản trị | [README](application/web/README.md) |
| [`detection-server-module`](detection-server-module) | API chat: nhận diện ảnh, chuẩn hóa câu hỏi, lưu hội thoại | [README](detection-server-module/README.md) |
| [`RAG-module`](RAG-module) | API tìm tài liệu và sinh câu trả lời | [README](RAG-module/README.md) |
| [`LLM-server-module`](LLM-server-module) | Chạy mô hình ngôn ngữ (Qwen 27B) bằng vLLM trên GPU thuê | [README](LLM-server-module/README.md) |

## Cần chuẩn bị

- Windows 10/11 (đã chạy thử), Git.
- **MongoDB Community Server** chạy ở `localhost:27017`.
- **Groq API key** ([console.groq.com](https://console.groq.com), gói miễn phí đủ dùng thử).
- Một LLM để sinh câu trả lời, chọn một:
  - vLLM trên GPU thuê ở Vast.ai (cấu hình chính), xem [LLM-server-module](LLM-server-module/README.md);
  - Gemini API key;
  - Ollama chạy trên máy (`ollama pull qwen3.5:4b`).
- Chạy bằng Docker: **Docker Desktop**. Chạy thủ công: **Python 3.11** và **Node.js 18+**.

GPU NVIDIA không bắt buộc; không có GPU thì RAG tính embedding bằng CPU (chậm hơn).

## Chạy nhanh bằng Docker

1. Tạo file cấu hình từ file mẫu (PowerShell, tại thư mục gốc repo):

   ```powershell
   Copy-Item .env.example .env
   Copy-Item RAG-module\.env.example RAG-module\.env
   Copy-Item detection-server-module\.env.example detection-server-module\.env
   ```

2. Điền các giá trị bắt buộc:

   | File | Biến |
   |---|---|
   | `detection-server-module/.env` | `GROQ_API_KEY`, `ADMIN_PASSWORD` (mật khẩu trang quản trị) |
   | `RAG-module/.env` | `LLM_PROVIDER` và cấu hình LLM đã chọn: `vllm` → `VLLM_API_KEY`; `gemini` → `GEMINI_API_KEY`; `ollama` → không cần thêm |
   | `.env` (gốc) | Chỉ khi dùng Vast: `VAST_SSH_KEY_FILE`. Máy không có GPU NVIDIA: bỏ `#` ở dòng `COMPOSE_FILE` |

3. Nếu dùng Vast: bật vLLM theo [LLM-server-module](LLM-server-module/README.md).
4. Nhấp đúp **`run.cmd`**. Khi được hỏi, dán lệnh SSH của Vast (nút **Connect**, dạng `ssh -p 41234 root@173.44.12.9 ...`); không dùng Vast thì bấm Enter.
5. Mở <http://localhost:5173>. Tắt bằng **`stop.cmd`**.

Lần đầu mất khoảng 10–30 phút: Docker tải thư viện (vài GB), RAG tải model embedding và tạo index, detection tải model nhận diện (khoảng 740 MB). Các lần sau chỉ mất khoảng một phút. Mỗi lần thuê GPU mới, Vast đổi IP và cổng: chạy lại `run.cmd` và dán lệnh SSH mới. Xem log, index lại, đổi cấu hình: [RUN_MODULES.md](RUN_MODULES.md).

## Chạy thủ công từng module

Mỗi module chạy trong một terminal riêng, theo thứ tự:

1. LLM: [vLLM trên Vast](LLM-server-module/README.md) và SSH tunnel, hoặc Gemini/Ollama.
2. [RAG module](RAG-module/README.md): `python -m server` (cổng 8010).
3. [Detection server](detection-server-module/README.md): `uvicorn app.api:app --port 8005`.
4. [Web](application/web/README.md): `npm run dev`, mở <http://localhost:5173>.

Mỗi README có phần cài đặt lần đầu. Lệnh bật hằng ngày gom trong [RUN_MODULES.md](RUN_MODULES.md).

## Cấu trúc thư mục

```text
application/web/           Web React + Vite
detection-server-module/   FastAPI: nhận diện ảnh, chat, quản trị
RAG-module/                FastAPI + CLI: tài liệu (data/), index, tìm kiếm, sinh câu trả lời
LLM-server-module/         Script cài và chạy vLLM trên Linux (Vast.ai)
docker/, docker-compose.yml, run.cmd, stop.cmd   Chạy cả hệ thống bằng Docker
```

## Kiểm thử

| Module | Lệnh (trong thư mục module, đã kích hoạt `.venv`) |
|---|---|
| RAG | `python -m pytest -q` |
| Detection | `python -m pytest -q tests` |
| LLM server | `python -m unittest discover -s tests -v` |

Các test chạy offline: không cần GPU, Groq, MongoDB hay LLM thật.

## Lưu ý

- Không có trong Git: các file `.env` (chứa key), trọng số model, index Chroma, `.venv`, `node_modules`. Chúng được tạo hoặc tải ở lần chạy đầu.
- Kết quả nhận diện và tư vấn chỉ mang tính tham khảo. Với bệnh nặng hoặc cây có giá trị cao, hãy hỏi thêm cán bộ kỹ thuật nông nghiệp.
