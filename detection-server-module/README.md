# PlantGPT - Plant Disease Chatbot

Chatbot tư vấn bệnh cây trồng bằng tiếng Việt. Ứng dụng kết hợp nhận diện ảnh lá cây, RAG (Retrieval-Augmented Generation), Groq LLM, lịch sử hội thoại MongoDB và Feedback RAG do quản trị viên duyệt.

## Chạy nhanh

Tạo `.env` với `GROQ_API_KEY` và `MONGO_URI` trước, theo [mẫu cấu hình](#cài-đặt-và-chạy) ở dưới. Sau đó chạy các lệnh PowerShell sau:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch torchvision ultralytics timm pillow requests huggingface-hub
python main.py
```

Mở <http://127.0.0.1:8005> để dùng chatbot.

> Kết quả nhận diện và nội dung tư vấn chỉ mang tính tham khảo. Với cây trồng có giá trị cao hoặc dấu hiệu bệnh nặng, hãy tham vấn cán bộ kỹ thuật nông nghiệp tại địa phương trước khi xử lý.

## Tính năng

- Nhận diện cây và bệnh từ ảnh lá qua ConvNeXt, GenYOLO và IEViT.
- Chuẩn hóa câu hỏi tiếng Việt, kể cả cách viết tắt hoặc lỗi chính tả, bằng Groq.
- Tìm kiếm kiến thức theo ngữ nghĩa từ ChromaDB trước khi tạo câu trả lời.
- Ghi nhớ ngữ cảnh nhận diện trong từng cuộc trò chuyện.
- Lưu cuộc hội thoại và đánh giá thích/không thích bằng MongoDB.
- Trang quản trị để chọn QA chất lượng, chỉnh câu trả lời và đưa vào Feedback RAG.
- Feedback RAG dùng embedding đa ngôn ngữ và cross-encoder reranker; chỉ các ví dụ phù hợp nhất được đưa vào prompt.
- Giao diện web chat và trang duyệt feedback có sẵn, không cần frontend build riêng.

## Luồng xử lý

```text
Tin nhắn + ảnh lá (tùy chọn)
        |
        +--> Chuẩn hóa câu hỏi (Groq)
        +--> Nhận diện ảnh: cây -> tách lá YOLO -> bệnh
        +--> Ghép ngữ cảnh: câu hỏi + ảnh + lịch sử phiên
        +--> Định tuyến: trả lời / yêu cầu ảnh / hỏi làm rõ / ngoài phạm vi
        +--> Domain RAG (ChromaDB) + Feedback RAG (MongoDB, tùy chọn)
        +--> Sinh câu trả lời tiếng Việt (Groq)
        +--> Lưu hội thoại và bản ghi feedback (MongoDB)
```

## Phạm vi dữ liệu

Kho kiến thức RAG hiện có 22 tài liệu về bệnh trên: táo, anh đào, ngô, nho, đào, ớt chuông, khoai tây, bí, dâu tây và cà chua.

Nhận diện bằng ảnh hiện hỗ trợ 9 nhóm cây có checkpoint bệnh tương ứng: táo, anh đào, ngô, nho, đào, ớt, khoai tây, dâu tây và cà chua. Dữ liệu về bí hiện dùng được cho câu hỏi văn bản qua RAG, nhưng chưa có checkpoint IEViT để chẩn đoán từ ảnh.

## Công nghệ

| Thành phần | Công nghệ |
| --- | --- |
| API và giao diện | FastAPI, HTML/CSS/JavaScript thuần |
| LLM | Groq API |
| Domain RAG | LangChain, ChromaDB, Hugging Face Embeddings |
| Feedback RAG | MongoDB, `multilingual-e5-small`, cross-encoder reranker |
| Nhận diện ảnh | ConvNeXt Tiny, GenYOLO (YOLOv8), IEViT |
| Lưu trữ | MongoDB |

## Yêu cầu

- Python 3.10 trở lên, khuyến nghị Python 3.11.
- MongoDB đang chạy cục bộ hoặc một MongoDB Atlas URI có thể kết nối.
- Groq API key.
- Internet ở lần chạy đầu để tải checkpoint thị giác và model embedding/reranker.
- RAM đủ cho các model AI; GPU CUDA là tùy chọn. Ứng dụng sẽ dùng CPU khi CUDA không khả dụng.

## Cài đặt và chạy

Các lệnh dưới đây dành cho PowerShell trên Windows.

```powershell
git clone <repository-url>
cd plant_disease_chatbot_RAG_complete_source

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Pipeline thị giác sử dụng PyTorch, Ultralytics, timm, Pillow, Requests và Hugging Face Hub. Nếu môi trường của bạn chưa có các gói này, cài thêm:

```powershell
# Chọn đúng lệnh PyTorch cho CPU/CUDA tại https://pytorch.org/get-started/locally/
pip install torch torchvision
pip install ultralytics timm pillow requests huggingface-hub
```

Tạo file `.env` tại thư mục gốc theo mẫu sau. Không commit file này.

```dotenv
GROQ_API_KEY=your_groq_api_key
NORMALIZER_MODEL=openai/gpt-oss-120b
ANSWER_MODEL=openai/gpt-oss-120b

MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=plant_chatbot

MAX_HISTORY_TURNS=12

# Feedback RAG
FEEDBACK_EMBEDDING_MODEL=intfloat/multilingual-e5-small
FEEDBACK_RERANKER_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
FEEDBACK_RETRIEVE_K=8
FEEDBACK_RERANK_K=2
FEEDBACK_SIMILARITY_THRESHOLD=0.55
FEEDBACK_RERANKER_ENABLED=true
```

Khởi động MongoDB, sau đó chạy ứng dụng:

```powershell
python main.py
```

Mở các địa chỉ sau:

- Chat: <http://127.0.0.1:8005>
- Duyệt feedback: <http://127.0.0.1:8005/admin>
- API docs: <http://127.0.0.1:8005/docs>
- Health check: <http://127.0.0.1:8005/health>

Lần khởi động đầu tiên sẽ kiểm tra và tải GenYOLO, một checkpoint ConvNeXt và các checkpoint IEViT theo từng loại cây vào `models/`. Thư mục này được `.gitignore` để tránh đẩy các file model lớn lên Git.

## Xây dựng lại chỉ mục RAG

Sau khi thêm hoặc sửa file `.txt` trong `rag/data/`, tạo lại ChromaDB:

```powershell
python -m rag.ingest --reset
```

`--reset` xóa chỉ mục cũ trong `rag/chroma_db/` trước khi lập chỉ mục lại, giúp tránh dữ liệu cũ bị giữ lại sau khi đổi tên hoặc xóa tài liệu. Bỏ `--reset` chỉ khi bạn thực sự muốn thêm tài liệu vào collection hiện có.

Tùy chỉnh RAG sau phải được đặt trong environment của tiến trình chạy ứng dụng (ví dụ PowerShell), vì module RAG đọc trực tiếp từ environment:

```powershell
$env:RAG_TOP_K = "5"
$env:RAG_CHUNK_SIZE = "1000"
$env:RAG_CHUNK_OVERLAP = "150"
$env:EMBEDDING_MODEL = "AITeamVN/Vietnamese_Embedding"
$env:EMBEDDING_DEVICE = "cpu"
python -m rag.ingest --reset
```

Có thể tắt ảnh preview từ bước tách lá hoặc điều chỉnh số lá được giữ lại:

```powershell
$env:SAVE_SEGMENTATION_PREVIEW = "false"
$env:YOLO_TOP_K = "2"
$env:YOLO_MIN_MASK_AREA = "100"
python main.py
```

Khi bật, preview được lưu ở `debug/segmentation/`; thư mục `debug/` cũng đã được bỏ qua bởi Git.

## API chính

| Method | Endpoint | Mô tả |
| --- | --- | --- |
| `GET` | `/health` | Kiểm tra API hoạt động. |
| `POST` | `/api/chat` | Gửi câu hỏi và ảnh tùy chọn dưới dạng `multipart/form-data`. |
| `GET` | `/api/conversations` | Liệt kê tối đa 100 cuộc trò chuyện gần nhất. |
| `GET` | `/api/conversations/{session_id}` | Lấy chi tiết hội thoại và trạng thái feedback. |
| `DELETE` | `/api/conversations/{session_id}` | Xóa một hội thoại. |
| `GET` | `/api/session/{session_id}` | Lấy snapshot phiên và kết quả nhận diện gần nhất. |
| `DELETE` | `/api/session/{session_id}` | Xóa dữ liệu phiên. |
| `PUT` | `/feedback` | Gửi đánh giá `like` hoặc `unlike`. |
| `GET` | `/admin/reviews` | Lấy các QA đã được người dùng đánh giá. |
| `POST` | `/admin/training` | Đưa các QA được chọn vào Feedback RAG. |
| `GET` | `/admin/training` | Xem các QA đang có trong Feedback RAG. |
| `DELETE` | `/admin/review/{feedback_id}` | Xóa feedback và vector Feedback RAG tương ứng. |

Ví dụ gửi câu hỏi kèm ảnh:

```powershell
curl.exe -X POST http://127.0.0.1:8005/api/chat `
  -F "session_id=demo-session-01" `
  -F "message=Lá cà chua này bị bệnh gì và xử lý thế nào?" `
  -F "image=@C:\duong-dan\la-ca-chua.jpg"
```

Trường `image` là tùy chọn. Khi chỉ hỏi về kiến thức đã biết, chỉ cần gửi `session_id` và `message`.

## Quy trình Feedback RAG

1. Mỗi câu trả lời của chatbot tạo một bản ghi feedback trong MongoDB.
2. Người dùng đánh giá thích hoặc không thích trên giao diện chat. Với `unlike`, hệ thống yêu cầu ít nhất một lý do.
3. Mở `/admin`, chọn các QA cần dùng làm dữ liệu hướng dẫn.
4. Với QA bị `unlike`, nhập câu trả lời đúng của quản trị viên trước khi index.
5. Nhấn nút index để lưu embedding và metadata vào collection `feedback_training_vectors`.
6. Ở các câu hỏi sau, hệ thống lọc theo cây/bệnh, tính cosine similarity, rerank rồi chỉ đưa các QA phù hợp vào LLM.

## Cấu trúc thư mục

```text
main.py                       Điểm khởi động Uvicorn
app/
  api.py                      FastAPI app, routes và dependency wiring
  chat/                       Chuẩn hóa, routing, context, session, pipeline
  llm/                        Groq adapter và interface LLM
  feedback/                   MongoDB feedback, session và Feedback RAG
  plant_ai/                   Nhận diện cây/bệnh và tách lá
  prompts/                    Prompt chuẩn hóa và sinh câu trả lời
  routes/                     API feedback và admin
  static/                     Giao diện chat và admin
rag/
  data/                       Tài liệu kiến thức bệnh cây (.txt)
  ingest.py                   Script tạo lại Chroma index
  retriever.py                Truy vấn ChromaDB
  service.py                  Adapter RAG cho pipeline chat
models/                       Checkpoint tải tự động, không commit
```

## Lưu ý khi đưa lên GitHub

- `.env`, `models/`, `rag/chroma_db/`, `debug/`, virtual environment và cache Python đã nằm trong `.gitignore`.
- Kiểm tra lại bằng `git status` trước khi push để chắc chắn không có API key, URI MongoDB có credential, ảnh người dùng hoặc checkpoint lớn.
- Cần có `GROQ_API_KEY` và MongoDB hợp lệ trước khi khởi động API; nếu thiếu, ứng dụng không thể khởi tạo các dịch vụ chat/lưu trữ.
- Chroma index và các model sẽ được tạo/tải ở máy triển khai. Không cần commit chúng.

## Kiểm tra nhanh

Sau khi server chạy, health check cần trả về:

```json
{"status":"ok"}
```

```powershell
curl.exe http://127.0.0.1:8005/health
```
