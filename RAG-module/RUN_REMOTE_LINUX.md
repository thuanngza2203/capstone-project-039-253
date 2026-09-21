# LLM remote trên Vast.ai, RAG ở host

Hướng dẫn và code server đã chuyển sang thư mục cùng cấp:
**[LLM-server-module/README.md](../LLM-server-module/README.md)**.

```text
Host: RAG-module, embedding, retrieval, Chroma, lịch sử chat
  → API qua SSH tunnel hoặc HTTPS
Vast.ai: LLM-server-module, vLLM, model trên GPU
```

Trên Vast, chỉ cài và chạy `LLM-server-module`. Trên host, ghép các giá trị
trong [rag-client.env.example](../LLM-server-module/rag-client.env.example)
vào `.env` RAG hiện có:

```dotenv
LLM_PROVIDER=vllm
VLLM_BASE_URL=http://127.0.0.1:8001/v1
VLLM_MODEL=qwen3.5-4b
VLLM_API_KEY=DIEN_CUNG_KEY_VOI_SERVER
VLLM_MAX_TOKENS=800
VLLM_TIMEOUT=120
VLLM_THINK=false
```

URL trên là ví dụ SSH tunnel `host:8001 → Vast:8000`. Với API public, dùng
URL HTTPS theo README server. Giữ các cấu hình embedding/index ở host và
khởi động lại RAG; không cần index lại chỉ vì đổi địa chỉ LLM.

Kiểm tra từ PowerShell trong `RAG-module`, sau khi activate venv:

```powershell
python ..\LLM-server-module\check_api.py --env-file .env
python main.py ask "Bệnh ghẻ trên cây táo xử lý như thế nào?"
python main.py chat --debug
```
