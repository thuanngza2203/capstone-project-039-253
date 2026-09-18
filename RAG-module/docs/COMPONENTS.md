# Các thành phần trong source

[← README](../README.md) · [Kiến trúc](ARCHITECTURE.md) · [Pipeline](PIPELINE.md)

## `config.py`

Module này đọc `.env`, khai báo đường dẫn và hằng số dùng chung, đồng thời chứa hai factory:

- Factory embedding tạo `HuggingFaceEmbeddings` với `AITeamVN/Vietnamese_Embedding`, vector được normalize và thiết bị mặc định là CPU.
- `create_chat_model()` chọn Ollama, Gemini hoặc vLLM qua `LLM_PROVIDER`.
- `get_retrieval_settings()` đọc mode, candidate-k, RRF và reranker; xem [RETRIEVAL.md](RETRIEVAL.md).
- `get_chat_settings()` đọc giới hạn lịch sử; xem [CHAT_MEMORY.md](CHAT_MEMORY.md).

Các biến môi trường được hỗ trợ:

| Biến | Mặc định | Dùng cho |
|---|---|---|
| `GEMINI_API_KEY` | Không có | Bắt buộc khi chọn provider Gemini |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Model sinh câu trả lời |
| `EMBEDDING_MODEL` | `AITeamVN/Vietnamese_Embedding` | Model tạo vector local |
| `EMBEDDING_DEVICE` | `cpu` | Thiết bị chạy embedding, ví dụ `cpu` hoặc `cuda` |

Các hằng số chính trong source là `CHUNK_SIZE=1000`, `CHUNK_OVERLAP=150`, `TOP_K=4`, đường dẫn `data/`, `chroma_db/` và tên Chroma collection.

## `rag.py`

Đây là module nghiệp vụ chính. API public:

### `load_documents() -> list[Document]`

Tìm `data/**/*.txt`, đọc UTF-8 và trả một `Document` cho mỗi file hợp lệ. Ngoài `source`, `crop`, `title`, loader tạo danh tính gồm tên bệnh chuẩn, `disease_id` ổn định và chuỗi `disease_aliases` phân cách bằng ` | `.

Lỗi nếu không có tài liệu hợp lệ hoặc nếu một file không giải mã được bằng UTF-8. Loader chỉ đọc mềm một số nhãn danh tính như TÊN TÀI LIỆU, TÊN BỆNH, Tên tiếng Anh và Tên gọi khác; các heading nội dung còn lại không bị ép theo một template cố định.

### `split_documents(documents) -> list[Document]`

Chia tài liệu bằng `RecursiveCharacterTextSplitter`. Separator ưu tiên ranh giới đoạn/câu trước khi phải cắt cứng. Metadata của tài liệu gốc được giữ trên mọi chunk.

Splitter bổ sung `start_index` để chỉ vị trí ký tự bắt đầu trong file gốc, sau đó prepend identity header `Tài liệu` / `Bệnh` / `Tên gọi` vào mọi chunk. Header trở thành một phần của nội dung được embedding và giúp các đoạn giữa file vẫn mang danh tính bệnh.

### `build_index() -> tuple[int, int]`

Load và split tài liệu, reset collection, tạo embedding rồi lưu vào Chroma. Kết quả là `(số tài liệu, số chunk)`.

Hàm luôn rebuild thay vì cập nhật tăng dần. Vì vậy cùng một corpus chạy `index` nhiều lần không tạo chunk trùng.

Identity header và metadata là một phần của index. Chỉ cần index lại khi đổi nội dung/header/embedding; bổ sung hybrid retrieval dùng tiếp index hiện có.

### `retrieve(question, k=TOP_K) -> list[Document]`

Kiểm tra query, mở Chroma, gọi `retrieval.search_store()` và trả top-k Document. Tham số `mode` chọn semantic, bm25 hoặc hybrid; `rerank` bật/tắt CrossEncoder. Mọi mode đều tìm toàn corpus, không khớp alias để tạo disease filter.

Hàm này không gọi LLM sinh câu trả lời. `retrieve_with_debug()` nhận cùng tham số và trả `SearchResult`: `.documents` để dùng tiếp, `.to_debug_dict()` để xem ứng viên/thứ hạng/điểm.

Chi tiết các hàm trong `retrieval.py`, cách inject store và đọc debug: [RETRIEVAL.md](RETRIEVAL.md).

Alias tiếp tục nằm trong identity header để hỗ trợ tìm kiếm; không có bảng cây/bệnh hay quy tắc phủ định để khóa phạm vi truy vấn.

### `format_context(documents) -> str`

Chuyển các chunk thành context có nhãn:

```text
[Nguồn 1: apple/apple_scab.txt]
Nội dung chunk...

[Nguồn 2: apple/apple_black_rot.txt]
Nội dung chunk...
```

Nhãn là cầu nối giữa metadata thật và citation mà prompt yêu cầu.

### `ask(question) -> tuple[str, list[str]]`

Gọi `retrieve()`, format context, chạy chuỗi `ChatPromptTemplate | chat model | StrOutputParser` và trả `(answer, source_paths)`.

`source_paths` được suy ra trực tiếp từ kết quả retrieval và loại bỏ đường dẫn lặp. Nếu không có chunk, hàm trả thông báo thiếu dữ liệu mà không gọi Gemini. Nếu thiếu API key hoặc API lỗi, hàm báo lỗi thay vì tự sinh một câu trả lời thay thế.

### `RAGSession`

Giữ Chroma, LLM client và lịch sử riêng cho một cuộc hội thoại trong cùng
process. `warmup()` chuẩn bị model local; `ask()` dùng history để viết query
độc lập, retrieve rồi trả lời; `ask_with_debug()` trả thêm query thực tế và
trace. `search()` luôn độc lập và không gọi LLM. `clear_history()` xóa lịch sử
mà không nạp lại tài nguyên.

## `conversation.py`

`ChatTurn` giữ câu gốc/query retrieval/câu trả lời. `ConversationMemory` giới hạn
số lượt và ký tự; `rewrite_question()` dùng cùng LLM của phiên để làm rõ câu
nối tiếp. Kiểm tra JSON đầu ra để tránh đưa câu trả lời vào retrieval. Xem
[CHAT_MEMORY.md](CHAT_MEMORY.md) để đọc luồng và các giới hạn.

## `main.py`

`argparse` ánh xạ bốn subcommand tới API trong `rag.py`:

```powershell
python main.py index
python main.py search "<câu hỏi>"
python main.py ask "<câu hỏi>"
python main.py chat --debug
```

CLI chỉ chịu trách nhiệm nhận input và trình bày output. Có thể import `rag.py` mà không phụ thuộc CLI.

## Các lớp LangChain được dùng

| Thành phần | Vai trò |
|---|---|
| `Document` | Gói nội dung văn bản và metadata đi cùng nhau |
| `RecursiveCharacterTextSplitter` | Chia văn bản theo đoạn/câu với overlap |
| `HuggingFaceEmbeddings` | Chạy embedding tiếng Việt trên máy |
| `Chroma` | Persist vector/text/metadata, semantic search và cung cấp text cho BM25 |
| `ChatPromptTemplate` | Ghép quy tắc, câu hỏi và context thành prompt |
| `ChatGoogleGenerativeAI` | Gọi Gemini để sinh câu trả lời |
| `StrOutputParser` | Chuyển message từ model thành chuỗi |

## Gọi API từ Python

```python
from rag import load_documents, split_documents

documents = load_documents()
chunks = split_documents(documents)
print(len(documents), len(chunks))
```

```python
from rag import retrieve

for doc in retrieve("Triệu chứng đốm mắt ếch là gì?", k=2):
    print(doc.metadata["source"])
    print(doc.page_content[:300])
```

```python
from rag import ask

answer, sources = ask("Quả khô trên cây táo liên quan đến bệnh nào?")
print(answer)
print("Nguồn:", *sources, sep="\n- ")
```

## Chọn LLM provider

Đổi `LLM_PROVIDER` trong `.env`, đặt model/endpoint/key tương ứng và chạy lại chương trình. Ví dụ:

```dotenv
LLM_PROVIDER=vllm
VLLM_BASE_URL=http://127.0.0.1:8000/v1
VLLM_MODEL=Qwen/Qwen3-0.6B
```

Không cần sửa `ask()` vì LangChain chat model vẫn tham gia cùng pipeline prompt/parser. Cũng không cần build lại Chroma nếu chỉ thay LLM. Nếu thay embedding model, bắt buộc chạy lại `python main.py index`.

## Vì sao V1 chưa có các thành phần khác?

- **Semantic chunking:** cần thêm embedding call trong ingestion và khó quan sát hơn recursive chunking.
- **Memory hội thoại:** làm câu hỏi phụ thuộc state, trong khi V1 cần mỗi truy vấn độc lập và dễ test.
- **Citation validator:** V1 đã trả danh sách source thật riêng; validation tự động có thể thêm khi cần độ tin cậy cao hơn.
- **PDF/OCR:** chất lượng extraction là một bài toán riêng; V1 yêu cầu người dùng chuẩn hóa về UTF-8 TXT trước.
