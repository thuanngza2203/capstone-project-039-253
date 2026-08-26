# Các thành phần trong source

[← README](../README.md) · [Kiến trúc](ARCHITECTURE.md) · [Pipeline](PIPELINE.md)

## `config.py`

Module này đọc `.env`, khai báo đường dẫn và hằng số dùng chung, đồng thời chứa hai factory:

- Factory embedding tạo `HuggingFaceEmbeddings` với `AITeamVN/Vietnamese_Embedding`, vector được normalize và thiết bị mặc định là CPU.
- `create_chat_model()` tạo `ChatGoogleGenerativeAI` với `gemini-2.5-flash`, `temperature=0` và tắt thinking budget cho pipeline đơn giản, ổn định.

Các biến môi trường được hỗ trợ:

| Biến | Mặc định | Dùng cho |
|---|---|---|
| `GEMINI_API_KEY` | Không có | Bắt buộc khi gọi `ask()` |
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

Identity header và các trường disease là một phần của index. Sau khi nâng cấp từ index cũ, phải chạy lại `python main.py index`.

### `retrieve(question, k=TOP_K) -> list[Document]`

Kiểm tra câu hỏi, mở index và dựng catalog bệnh từ metadata. Nếu query khớp duy nhất một `disease_id`, hàm truyền Chroma filter theo ID đó; nếu không rõ hoặc nhắc nhiều bệnh, hàm dense search toàn collection. Kết quả trong phạm vi tìm kiếm được xếp theo độ tương đồng, tối đa `k` document.

Hàm này không gọi Gemini nên phù hợp để debug retrieval hoặc dùng trong code khác.

Filter chỉ bật khi việc đối chiếu tên bệnh/tên gọi khác cho đúng một ID. Câu hỏi mô tả triệu chứng không có tên bệnh và câu hỏi so sánh từ hai bệnh trở lên cố ý không filter.

Tên được so khớp không phân biệt chữ hoa hay dấu tiếng Việt. Các từ cấu trúc như “bệnh”, “trên”, “cây” được bỏ qua, nhưng các từ còn lại của alias phải tạo thành một cụm liên tiếp. Vì vậy “bệnh ghẻ trên táo” vẫn khớp, còn mô tả “quả bị thối rồi chuyển màu đen” không tự động bị coi là tên bệnh thối đen. Cụm đứng sau phủ định rõ như “không phải” hoặc “loại trừ” cũng không bật filter.

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

## `main.py`

`argparse` ánh xạ ba subcommand tới API trong `rag.py`:

```powershell
python main.py index
python main.py search "<câu hỏi>"
python main.py ask "<câu hỏi>"
```

CLI chỉ chịu trách nhiệm nhận input và trình bày output. Có thể import `rag.py` mà không phụ thuộc CLI.

## Các lớp LangChain được dùng

| Thành phần | Vai trò |
|---|---|
| `Document` | Gói nội dung văn bản và metadata đi cùng nhau |
| `RecursiveCharacterTextSplitter` | Chia văn bản theo đoạn/câu với overlap |
| `HuggingFaceEmbeddings` | Chạy embedding tiếng Việt trên máy |
| `Chroma` | Persist vector/identity metadata, similarity search local và filter theo `disease_id` |
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

## Thay Gemini bằng local LLM sau này

Giữ nguyên chữ ký `create_chat_model()` và chỉ đổi phần khởi tạo bên trong `config.py`. Ví dụ định hướng với Ollama sau khi bổ sung dependency phù hợp:

```python
from langchain_ollama import ChatOllama


def create_chat_model():
    return ChatOllama(model="ten-model-local", temperature=0)
```

Không cần sửa `ask()` vì LangChain chat model vẫn tham gia cùng pipeline prompt/parser. Cũng không cần build lại Chroma nếu chỉ thay LLM. Nếu thay embedding model, bắt buộc chạy lại `python main.py index`.

## Vì sao V1 chưa có các thành phần khác?

- **BM25/hybrid search và reranker:** tăng dependency và nhiều tham số phải đánh giá; dense search đủ để tạo baseline dễ hiểu.
- **Semantic chunking:** cần thêm embedding call trong ingestion và khó quan sát hơn recursive chunking.
- **Memory hội thoại:** làm câu hỏi phụ thuộc state, trong khi V1 cần mỗi truy vấn độc lập và dễ test.
- **Citation validator:** V1 đã trả danh sách source thật riêng; validation tự động có thể thêm khi cần độ tin cậy cao hơn.
- **PDF/OCR:** chất lượng extraction là một bài toán riêng; V1 yêu cầu người dùng chuẩn hóa về UTF-8 TXT trước.
