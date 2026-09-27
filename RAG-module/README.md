# RAG module

Tìm đoạn tài liệu liên quan tới câu hỏi và sinh câu trả lời có dẫn nguồn. Dùng qua **API** (detection server và web gọi) hoặc **CLI** (thử nhanh trong terminal).

- Kho tài liệu: `data/` gồm 25 file `.txt`, mỗi file một bệnh, chia mục theo tiêu đề.
- Chia chunk theo tiêu đề (tối đa 400 token), embedding `AITeamVN/Vietnamese_Embedding`, lưu trong ChromaDB.
- Tìm lai: tìm theo ngữ nghĩa + BM25, gộp bằng RRF; gửi 6 chunk tốt nhất cho LLM.
- LLM chọn bằng `LLM_PROVIDER`: `vllm` (Qwen 27B trên Vast.ai), `gemini` hoặc `ollama`.

## Cài đặt lần đầu

Cần Python 3.11. Lần chạy đầu cần Internet để tải model embedding (khoảng 2,3 GB).

```powershell
cd RAG-module
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Có GPU NVIDIA: trên Windows, lệnh trên cài PyTorch bản CPU. Muốn chạy embedding trên GPU thì cài PyTorch bản CUDA theo [pytorch.org](https://pytorch.org/get-started/locally/) trước khi cài `requirements.txt`, rồi đặt `EMBEDDING_DEVICE=cuda`. Không có GPU thì để `EMBEDDING_DEVICE=cpu`.

PowerShell chặn `Activate.ps1`: chạy `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` rồi activate lại.

## Cấu hình `.env`

Chọn một LLM và điền phần tương ứng:

| `LLM_PROVIDER` | Cần điền | Ghi chú |
|---|---|---|
| `vllm` | `VLLM_API_KEY` (trùng `LLM_API_KEY` trên server), `VLLM_MODEL=rag-llm` | Server và SSH tunnel theo [LLM-server-module](../LLM-server-module/README.md); chép các dòng trong `rag-client.env.example` của module đó |
| `gemini` | `GEMINI_API_KEY`, `GEMINI_MODEL` | Chỉ câu hỏi và các chunk được gửi đi |
| `ollama` | `OLLAMA_MODEL` (mặc định `qwen3.5:4b`) | Cài [Ollama](https://ollama.com/download) rồi `ollama pull qwen3.5:4b` |

Các biến khác đã có giá trị mặc định hợp lý trong `.env.example`:

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `EMBEDDING_DEVICE` | `cuda` | `cpu` nếu không có GPU NVIDIA |
| `RAG_TOP_K` | `6` | Số chunk gửi cho LLM |
| `RETRIEVAL_MODE` | `hybrid` | `semantic`, `bm25` hoặc `hybrid` |
| `CHUNKING_STRATEGY` | `structure` | `structure` (theo tiêu đề) hoặc `recursive` (theo số ký tự) |
| `RAG_API_KEY` | trống | Đặt key thì client phải gửi `Authorization: Bearer <key>`; detection dùng cùng key |

Biến môi trường của terminal được ưu tiên hơn `.env`. Sửa `.env` xong phải khởi động lại chương trình.

## Tạo index

Chạy một lần, và chạy lại khi sửa tài liệu trong `data/`, đổi cách chia chunk hoặc model embedding:

```powershell
python main.py index
```

Kết quả: 25 tài liệu thành 743 chunk trong `chroma_db_structure/`. Đổi LLM, số chunk hay cách tìm thì **không** cần index lại.

## Chạy API

```powershell
python -m server
```

Swagger: <http://127.0.0.1:8010/docs>. Các endpoint chính:

| Endpoint | Việc | Gọi LLM |
|---|---|---|
| `GET /health` | Server còn sống | Không |
| `GET /v1/status` | Index, số chunk, LLM đang dùng | Không |
| `GET /v1/llm?probe=true` | Hỏi server LLM model nào đang thật sự chạy (kiểm tra kết nối) | Không |
| `POST /v1/retrieve` | Tìm chunk theo câu hỏi, có thể khoanh theo `plant_type`, `disease` | Không |
| `POST /v1/answer` | Tìm chunk và sinh câu trả lời có dẫn nguồn | Có |
| `GET /v1/taxonomy` | Danh sách cây, bệnh và tên gọi khác | Không |
| `GET /v1/admin/...` | Xem tài liệu, chunk cho trang quản trị | Không |

## Dùng CLI

```powershell
python main.py search "Bệnh ghẻ táo có triệu chứng gì?" --debug   # chỉ tìm, không gọi LLM
python main.py ask "Cách quản lý bệnh thối đen trên táo?"         # tìm và trả lời
python main.py chat                                              # hỏi liên tục, nhớ lịch sử
python main.py preview-chunks --source apple/apple_scab.txt      # xem cách chia chunk
```

Trong `chat`: `/reset` xóa lịch sử, `/exit` để thoát.

## Kiểm thử

```powershell
python -m pytest -q
```

Test chạy offline, không cần GPU, LLM hay tải model.

## Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| API báo index chưa sẵn sàng | Chạy `python main.py index` |
| `Connection refused` khi hỏi | LLM chưa chạy: kiểm tra Ollama (`ollama list`), hoặc SSH tunnel tới Vast |
| vLLM báo 401 | `VLLM_API_KEY` phải trùng `LLM_API_KEY` trên server |
| vLLM báo 404 model | `VLLM_MODEL` phải trùng `LLM_SERVED_MODEL_NAME` (mặc định `rag-llm`) |
| Báo điền cả hai cách khai báo địa chỉ | Dùng `VLLM_HOST` + `VLLM_PORT`, **hoặc** `VLLM_BASE_URL`, không dùng cả hai |
| Chạy chậm, hết VRAM | Đặt `EMBEDDING_DEVICE=cpu` |

## Cấu trúc

```text
main.py           CLI: index, search, ask, chat, preview-chunks
server/           API FastAPI (python -m server)
config.py         Đọc .env
chunking.py       Chia chunk theo tiêu đề
rag.py            Index, tìm kiếm, sinh câu trả lời
retrieval.py      Tìm theo ngữ nghĩa, BM25, RRF, xếp hạng lại
taxonomy.py       Bảng tên cây, bệnh và tài liệu tương ứng
data/             Tài liệu bệnh cây (.txt)
eval/             Bộ 60 câu hỏi đánh giá truy xuất
experiments/      Thí nghiệm so sánh model, khảo sát người dùng
docs/             Tài liệu kỹ thuật chi tiết (SETUP, ARCHITECTURE, DATA_GUIDE...)
tests/            Test offline
```

Thêm tài liệu mới: xem [docs/DATA_GUIDE.md](docs/DATA_GUIDE.md), rồi chạy lại `python main.py index`.
