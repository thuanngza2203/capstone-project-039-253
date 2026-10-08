# LLM server (vLLM trên Vast.ai)

Chạy mô hình ngôn ngữ sinh câu trả lời trên GPU thuê ở [Vast.ai](https://vast.ai), bằng [vLLM](https://docs.vllm.ai). API tương thích OpenAI, chỉ nghe ở `127.0.0.1:8000` trên máy Vast. RAG module trên máy bạn gọi tới qua SSH tunnel.

- Model mặc định: `Qwen/Qwen3.8-27B-FP8`, gọi qua tên `rag-llm`.
- Cần GPU NVIDIA khoảng 48 GB VRAM (ví dụ RTX 4090 48 GB, A6000). Model nhỏ hơn thì GPU nhỏ hơn.
- Không dùng Vast: RAG module chạy được với Gemini hoặc Ollama, bỏ qua module này.

## 1. Cài trên Vast (một lần cho mỗi instance)

Thuê instance Ubuntu có GPU NVIDIA, thêm SSH public key vào tài khoản Vast, rồi SSH vào (lệnh lấy ở nút **Connect**). Trong terminal Linux:

```bash
apt-get update && apt-get install -y git python3 python3-venv python3-pip tmux nano build-essential ca-certificates
nvidia-smi                       # phải thấy GPU

mkdir -p /workspace && cd /workspace
git clone --filter=blob:none --sparse https://github.com/thuanngza2203/capstone-project-039-253.git
cd capstone-project-039-253
git sparse-checkout set LLM-server-module
cd LLM-server-module
LLM_API_KEY=<key> bash install.sh   # tạo .venv, cài vLLM, tạo .env với key này
```

`<key>` là giá trị `VLLM_API_KEY` đang có trong `RAG-module/.env` trên máy bạn, để hai bên khớp nhau. Bỏ `LLM_API_KEY=<key>` thì script tự sinh key ngẫu nhiên và in ra; chép key đó vào `VLLM_API_KEY` của RAG. Đã có `.env` thì script giữ nguyên, không đổi key.

Máy Vast đã clone từ trước: `git pull` rồi chạy lại `bash install.sh`, không cần xoá `.venv`.

Các biến chính trong `.env` (sửa bằng `nano .env`; `Ctrl+O`, Enter để lưu, `Ctrl+X` để thoát):

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `LLM_MODEL_ID` | `Qwen/Qwen3.8-27B-FP8` | Model trên HuggingFace |
| `LLM_SERVED_MODEL_NAME` | `rag-llm` | Tên RAG dùng khi gọi (`VLLM_MODEL`); đổi model vẫn giữ tên này |
| `LLM_API_KEY` | | Bắt buộc, trùng `VLLM_API_KEY` bên RAG |
| `LLM_MAX_MODEL_LEN` | `16384` | Số token tối đa mỗi request (câu hỏi, lịch sử, tài liệu, câu trả lời) |
| `LLM_MAX_NUM_SEQS` | `4` | Số request xử lý song song |
| `LLM_GPU_MEMORY_UTILIZATION` | `0.90` | Tỉ lệ VRAM dành cho vLLM |

## 2. Bật server

Trong tmux (thoát SSH không làm tắt server; `Ctrl+B` rồi `D` để thoát tạm, `tmux attach` để vào lại):

```bash
cd /workspace/capstone-project-039-253/LLM-server-module
source .venv/bin/activate
python serve.py --dry-run        # kiểm tra cấu hình, in lệnh sẽ chạy
python serve.py                  # lần đầu tải model (khoảng 30 GB), các lần sau dùng lại cache
```

Ở cửa sổ tmux khác, khi server đã sẵn sàng:

```bash
python check_api.py              # thành công: "API hoạt động, model: rag-llm" kèm một câu trả lời
```

## 3. Kết nối từ máy bạn

Mỗi lần thuê GPU mới, Vast cấp IP và cổng SSH mới. Lấy lệnh SSH ở nút **Connect**.

**Chạy bằng Docker:** nhấp đúp `run.cmd` ở thư mục gốc repo, dán lệnh SSH đó. Tunnel được tạo tự động. Xem [README gốc](../README.md).

**Chạy thủ công:** mở tunnel trong một PowerShell riêng và giữ cửa sổ này mở (thay `<PORT>`, `<IP>` theo lệnh của Vast):

```powershell
ssh -i "$env:USERPROFILE\.ssh\vast_ed25519" -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -p <PORT> -L 127.0.0.1:8001:127.0.0.1:8000 root@<IP>
```

Rồi chép các dòng trong [rag-client.env.example](rag-client.env.example) vào `RAG-module/.env`, điền key:

```dotenv
LLM_PROVIDER=vllm
VLLM_BASE_URL=http://127.0.0.1:8001/v1
VLLM_HOST=
VLLM_PORT=
VLLM_MODEL=rag-llm
VLLM_API_KEY=<key đã tạo ở bước 1>
```

Kiểm tra từ máy bạn: `python ..\LLM-server-module\check_api.py --env-file .env` (chạy trong `RAG-module`, đã activate `.venv`).

## Đổi model

Sửa `LLM_MODEL_ID` trong `.env`, giữ `LLM_SERVED_MODEL_NAME=rag-llm`, rồi `Ctrl+C` server cũ và chạy lại `python serve.py`. Model khác họ Qwen cần xem lại `LLM_REASONING_PARSER` và `LLM_CHAT_TEMPLATE_KWARGS` (ghi chú trong `.env.example`). Đổi model không cần index lại tài liệu.

## Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `bash install.sh` báo `invalid peer certificate: UnknownIssuer` | Máy Vast đi qua proxy HTTPS có CA riêng: pip tin được (dùng kho chứng chỉ của hệ điều hành), uv thì không. Script đã tự cho uv dùng kho đó (dòng `uv dung kho chung chi cua he dieu hanh` trong log); bản cũ chưa có thì `git pull` rồi chạy lại. Vẫn lỗi: kiểm tra `ca-certificates` đã cài; lỗi riêng ở `download.pytorch.org` thì chạy `LLM_TORCH_BACKEND=pypi bash install.sh`. Máy có mirror pip (`PIP_INDEX_URL` hoặc cấu hình pip) thì uv tự dùng chung |
| `serve.py` báo `Can't load tokenizer for 'Qwen/...'` | Máy Vast không tải được model từ HuggingFace. Lỗi chứng chỉ (`CERTIFICATE_VERIFY_FAILED`) do proxy HTTPS: `serve.py` đã tự trỏ `SSL_CERT_FILE` về kho chứng chỉ của hệ điều hành (dòng `Chứng chỉ TLS khi tải model` khi khởi động). Lỗi gốc có `CAS Client Error ... 401` (xethub): thêm `HF_HUB_DISABLE_XET=1` vào `.env`. Không kết nối được HuggingFace: thêm `HF_ENDPOINT=https://hf-mirror.com`. Vẫn lỗi thì thuê máy ở khu vực khác |
| `Connection refused` | Server chưa sẵn sàng, tunnel chưa mở, hoặc IP/cổng SSH đã đổi (thuê máy mới) |
| HTTP 401/403 | `VLLM_API_KEY` bên RAG phải trùng `LLM_API_KEY` |
| HTTP 404, sai model | `VLLM_MODEL` phải trùng `LLM_SERVED_MODEL_NAME`; URL phải kết thúc bằng `/v1` |
| Hết bộ nhớ GPU (OOM) | Giảm `LLM_MAX_MODEL_LEN` hoặc `LLM_MAX_NUM_SEQS`, hoặc chọn model nhỏ hơn |
| Sửa `.env` không có tác dụng | Dừng và chạy lại `python serve.py` |

Tắt máy Vast: thoát terminal **không** dừng tính phí. Stop vẫn tính phí ổ đĩa; Destroy xóa hẳn instance.

## Các file

| File | Công dụng |
|---|---|
| `install.sh` | Tạo `.venv`, cài vLLM, kiểm tra CUDA, tạo `.env` lần đầu (`bash install.sh --upgrade` để nâng cấp, `--help` xem tùy chọn) |
| `serve.py` | Đọc `.env`, khởi động vLLM; tải model bằng kho chứng chỉ của hệ điều hành |
| `check_api.py` | Kiểm tra kết nối, API key và một câu trả lời |
| `rag-client.env.example` | Các dòng cần chép sang `.env` của RAG module |
| `nginx.conf.example` | Tùy chọn: mở API qua HTTPS có tên miền thay cho SSH tunnel |
| `tests/` | Test offline: `python -m unittest discover -s tests -v` |
