# PlantGPT - Plant Disease Chatbot

Chatbot tư vấn bệnh cây trồng bằng tiếng Việt. Ứng dụng kết hợp nhận diện ảnh lá cây, RAG (Retrieval-Augmented Generation), Groq LLM, lịch sử hội thoại MongoDB và Feedback RAG do quản trị viên duyệt.

## Chạy nhanh

Chép `.env.example` thành `.env`, điền `GROQ_API_KEY` và `MONGO_URI`. Sau đó chạy các lệnh PowerShell sau:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env   # rồi điền key
python main.py
```

Mặc định `ANSWER_BACKEND=groq`: trả lời bằng `rag/` nội bộ + Groq như trước. Lần đầu cần
tạo chỉ mục: `python -m rag.ingest --reset`.

Mở <http://127.0.0.1:8005> để dùng chatbot.

> Kết quả nhận diện và nội dung tư vấn chỉ mang tính tham khảo. Với cây trồng có giá trị cao hoặc dấu hiệu bệnh nặng, hãy tham vấn cán bộ kỹ thuật nông nghiệp tại địa phương trước khi xử lý.

### Linux (venv)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # điền GROQ_API_KEY, MONGO_URI, ADMIN_PASSWORD
python -m rag.ingest --reset  # lần đầu, tạo chỉ mục RAG nội bộ
uvicorn app.api:app --host 0.0.0.0 --port 8005
```

`python main.py` cũng chạy được nhưng chỉ nghe `127.0.0.1` và bật `reload` (dành cho dev).

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
        +--> (chỉ ACCEPT_QUERY) dựng RagAnswerRequest -> AnswerBackend
        |       ANSWER_BACKEND=rag : POST {RAG_API_URL}/v1/answer (RAG-module)
        |       ANSWER_BACKEND=groq: rag/ nội bộ + Feedback RAG + Groq (như cũ)
        +--> Lưu hội thoại và bản ghi feedback (MongoDB)
```

## Nguồn câu trả lời: `ANSWER_BACKEND`

Pipeline chỉ dựng một payload đúng hợp đồng
[`../2026-09-23-rag-api-openapi.json`](../2026-09-23-rag-api-openapi.json) rồi giao cho
`AnswerBackend` (`app/answer/`). Đổi backend chỉ là đổi `.env`, không sửa code:

| Biến | Ý nghĩa |
| --- | --- |
| `ANSWER_BACKEND` | `groq` (mặc định) = luồng cũ; `rag` = gọi RAG server theo file OpenAPI (server do bên RAG cung cấp). |
| `RAG_API_URL` | Địa chỉ RAG server, mặc định `http://127.0.0.1:8010`. |
| `RAG_API_KEY` | Khớp `RAG_API_KEY` bên RAG-module; gửi qua `Authorization: Bearer`. |
| `RAG_API_TIMEOUT` | Giây, mặc định 150 (lớn hơn timeout LLM 120 bên RAG). |
| `RAG_SEARCH_ORIGINAL_QUERY` | `false` (mặc định); `true` = gửi kèm câu gốc trong `extra_queries`. Đo 24/09 không thấy lợi. |

Payload gửi sang RAG:

| Trường | Lấy từ |
| --- | --- |
| `query` | Câu gốc của người dùng (hoặc `"Ảnh này đang bị bệnh gì?"` khi chỉ gửi ảnh). |
| `retrieval_query` | Câu Groq đã chuẩn hóa (`RetrievalQueryBuilder`). Cây/bệnh lấy từ ảnh hoặc lượt trước không gắn vào câu: `plant_type`/`disease` đã khoanh đúng tài liệu. |
| `extra_queries` | Chỉ khi `RAG_SEARCH_ORIGINAL_QUERY=true`: `[câu gốc]` khi khác câu chuẩn hóa; RAG tìm cả hai câu rồi gộp bằng RRF. |
| `plant_type`, `disease` | Nguyên giá trị đã resolve (ví dụ `Apple`, `Apple___Apple_scab`). Bệnh Groq chỉ đoán từ cách gọi chung chung (`disease_named=false`) không được gửi: RAG tìm theo cây. |
| `history` | Tối đa 12 message gần nhất `{role, content}`, mỗi content ≤ 8000 ký tự. |
| `subject_context` | Kết quả nhận diện dạng chữ, khi có ảnh hoặc câu hỏi nối tiếp. |
| `rewrite_query` | Chỉ `true` khi Groq lỗi: không có câu chuẩn hóa, RAG tìm bằng câu gốc và tự viết lại câu nối tiếp. |

Groq lỗi (hết quota, mạng, JSON sai) không làm `/api/chat` trả 500: câu gốc được gửi thẳng
sang RAG, chỉ kèm cây/bệnh từ ảnh của lượt này, và `debug.normalizer_failed=true`.

`sources`, `scope_status`, `grounded` được lưu vào `metadata` của bản ghi feedback và
trả thêm trong `POST /api/chat` (trường `sources`). Khi RAG lỗi, route trả `503`
(key sai, LLM lỗi, index chưa sẵn sàng, không kết nối được) hoặc `500` (payload sai
hợp đồng) kèm thông báo tiếng Việt, và **không lưu** nửa lượt chat.

Feedback RAG: với `ANSWER_BACKEND=rag`, tối đa 3 câu trả lời admin đã duyệt (lọc theo cây/bệnh,
cosine rồi rerank như backend groq) được gửi kèm trong `feedback_examples`; RAG đưa vào prompt và
LLM ưu tiên mẫu cùng ý. Feedback RAG lỗi (Mongo, tải model) thì vẫn trả lời, chỉ không có mẫu.

**Tìm trên web** (`web_search=true`): sau bước nhận diện ảnh, Groq (`WEB_SEARCH_MODEL`, tool
`browser_search`) tự tìm web và trả lời; không qua normalizer, router hay RAG. Lịch sử gần nhất và
kết quả nhận diện (ảnh lượt này hoặc ảnh gần nhất) đi kèm câu hỏi. Trang web model đã đọc nằm ở
`web_sources`; `action=WEB_SEARCH`. Groq lỗi thì trả 503 và không lưu nửa lượt chat.

## Phạm vi dữ liệu

Kho kiến thức RAG hiện có 22 tài liệu về bệnh trên: táo, anh đào, ngô, nho, đào, ớt chuông, khoai tây, bí, dâu tây và cà chua.

Ảnh không có lá cây (người, đồ vật, ảnh chụp màn hình…) bị từ chối ngay với thông báo "Ảnh không hợp lệ" (HTTP 400); ngưỡng chỉnh bằng `LEAF_MIN_CONFIDENCE`, `LEAF_MIN_AREA_RATIO`.

Nhận diện bằng ảnh hiện hỗ trợ 9 nhóm cây có checkpoint bệnh tương ứng: táo, anh đào, ngô, nho, đào, ớt, khoai tây, dâu tây và cà chua. Dữ liệu về bí hiện dùng được cho câu hỏi văn bản qua RAG, nhưng chưa có checkpoint IEViT để chẩn đoán từ ảnh. Ảnh được phân loại là cây không có checkpoint bệnh (cam, bí) vẫn trả `200`: kết quả nhận diện chỉ có tên cây, `disease=null`.

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

`requirements.txt` đã gồm cả pipeline thị giác (PyTorch, Ultralytics, timm, Pillow,
Requests, Hugging Face Hub). Muốn chọn bản PyTorch CPU/CUDA cụ thể thì cài torch
trước theo ghi chú ở đầu `requirements.txt`.

Chép `.env.example` thành `.env` rồi điền `GROQ_API_KEY`, `MONGO_URI` và
`ADMIN_PASSWORD` (đăng nhập trang `/admin`). Chỉ khi dùng `ANSWER_BACKEND=rag` mới cần
`RAG_API_URL`, `RAG_API_KEY`. Không commit `.env`. Mọi biến trong `.env` đều có hiệu lực, kể cả các
biến đọc bằng `os.getenv` (`YOLO_*`, `RAG_*`, `SAVE_SEGMENTATION_PREVIEW`...).

Khởi động MongoDB, sau đó chạy ứng dụng:

```powershell
python main.py
```

Mở các địa chỉ sau:

- Chat: <http://127.0.0.1:8005>
- Duyệt feedback: <http://127.0.0.1:8005/admin> (trình duyệt hỏi `ADMIN_USERNAME` / `ADMIN_PASSWORD`)
- API docs: <http://127.0.0.1:8005/docs>
- Health check: <http://127.0.0.1:8005/health>

Lần khởi động đầu tiên sẽ kiểm tra và tải GenYOLO, một checkpoint ConvNeXt và các checkpoint IEViT theo từng loại cây vào `models/` của module (tính theo thư mục module, chạy từ đâu cũng dùng chung). Thư mục này được `.gitignore` để tránh đẩy các file model lớn lên Git.

## Xây dựng lại chỉ mục RAG nội bộ (chỉ `ANSWER_BACKEND=groq`)

Với `ANSWER_BACKEND=rag`, tài liệu và index nằm bên RAG server; `rag/` ở đây không được dùng.

Với `ANSWER_BACKEND=groq`, sau khi thêm hoặc sửa file `.txt` trong `rag/data/`, tạo lại ChromaDB:

```powershell
python -m rag.ingest --reset
```

`--reset` xóa chỉ mục cũ trong `rag/chroma_db/` trước khi lập chỉ mục lại, giúp tránh dữ liệu cũ bị giữ lại sau khi đổi tên hoặc xóa tài liệu. Bỏ `--reset` chỉ khi bạn thực sự muốn thêm tài liệu vào collection hiện có.

Tùy chỉnh `RAG_TOP_K`, `RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP`, `EMBEDDING_MODEL`,
`EMBEDDING_DEVICE` đặt trong `.env` (hoặc environment của tiến trình, được ưu tiên hơn).

Tương tự, `SAVE_SEGMENTATION_PREVIEW`, `YOLO_TOP_K`, `YOLO_MIN_MASK_AREA` điều khiển bước
tách lá. Khi bật preview, ảnh được lưu ở `debug/segmentation/`; thư mục `debug/` đã được bỏ qua bởi Git.

## API chính

| Method | Endpoint | Mô tả |
| --- | --- | --- |
| `GET` | `/health` | Kiểm tra API hoạt động. |
| `GET` | `/api/models` | Model trả lời người dùng chọn được (`CHAT_LLM_PROVIDERS`, rỗng khi `ANSWER_BACKEND=groq`) và có nút Tìm trên web không (`web_search`). |
| `POST` | `/api/chat` | Gửi câu hỏi, ảnh tùy chọn, `llm_provider` hoặc `web_search=true` dưới dạng `multipart/form-data`. |
| `POST` | `/admin/login` | Đăng nhập trang admin của web, trả token dùng trong `Authorization: Bearer`. |
| `GET` | `/api/conversations` | Liệt kê tối đa 100 cuộc trò chuyện gần nhất. |
| `GET` | `/api/conversations/{session_id}` | Lấy chi tiết hội thoại và trạng thái feedback. |
| `DELETE` | `/api/conversations/{session_id}` | Xóa một hội thoại. |
| `GET` | `/api/session/{session_id}` | Lấy snapshot phiên và kết quả nhận diện gần nhất. |
| `DELETE` | `/api/session/{session_id}` | Xóa dữ liệu phiên. |
| `PUT` | `/feedback` | Gửi đánh giá `like` hoặc `unlike`. |
| `GET` | `/admin/status` | Backend trả lời đang dùng, ví dụ Feedback RAG có được dùng không. |
| `GET` | `/admin/reviews` | Lấy các QA đã được người dùng đánh giá. |
| `POST` | `/admin/training` | Đưa các QA được chọn vào Feedback RAG. |
| `GET` | `/admin/training` | Xem các QA đang có trong Feedback RAG. |
| `DELETE` | `/admin/review/{feedback_id}` | Xóa feedback và vector Feedback RAG tương ứng. |

Mọi route `/admin` và `/admin/*` (trừ `/admin/login`) cần đăng nhập khi `ADMIN_PASSWORD` có giá trị:
HTTP Basic (trang admin cũ) hoặc token của `/admin/login` (web). Web bắt buộc có `ADMIN_PASSWORD` mới đăng nhập được.

Câu trả lời gửi về người dùng đã bỏ nhãn `[Nguồn n]` (RAG vẫn yêu cầu LLM ghi để đo trích dẫn);
tài liệu đã dùng nằm ở `source_documents`: tên tài liệu, file trong kho và link nguồn tham khảo.
`MAX_HISTORY_TURNS` (mặc định 6) là số message gần nhất đưa vào normalizer và answer LLM.

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

## Chạy test

Test chạy offline (không cần Groq, MongoDB, RAG server hay checkpoint):

```powershell
python -m pytest -q tests
```

## Cấu trúc thư mục

```text
main.py                       Điểm khởi động Uvicorn
app/
  api.py                      FastAPI app, routes và dependency wiring
  answer/                     AnswerBackend: GroqAnswerBackend, RagHttpBackend
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
tests/                        Test offline
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
