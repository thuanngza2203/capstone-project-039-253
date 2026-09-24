# Changelog

## 2026-09-24 — Câu tìm tài liệu dùng câu Groq đã chuẩn hóa

- **Sửa lỗi làm mọi lượt chat trả 500:** normalizer gửi `reasoning_effort="none"`, nhưng
  `openai/gpt-oss-120b` (model mặc định trong `.env`) chỉ nhận `low`/`medium`/`high` nên Groq trả
  400 ở mọi lần gọi. Thêm `NORMALIZER_REASONING_EFFORT` (mặc định `low`; qwen/qwen3-32b dùng `none`).
- **Câu tìm (`retrieval_query`) là câu Groq đã chuẩn hóa**, không còn câu mẫu theo intent ("Cách
  điều trị bệnh apple_scab trên cây apple"). Câu mẫu làm rơi chi tiết người dùng hỏi: đo trên 58 câu
  có nhiễu, Recall@4 0,138 (câu mẫu) → 0,793 (câu chuẩn hóa). Cây/bệnh lấy từ ảnh hoặc lượt trước
  chỉ đi trong `plant_type`/`disease`, không gắn vào câu: đo cho thấy gắn "(bệnh …, cây …)" làm
  Hit@1 giảm 0,649 → 0,491. Chi tiết: `RAG-module/reports/2026-09-24-query-normalization/`.
- **Tùy chọn gửi kèm câu gốc** trong `extra_queries` (RAG API 1.2.0 tìm cả hai câu, gộp bằng RRF):
  `RAG_SEARCH_ORIGINAL_QUERY`, **tắt mặc định** vì đo không thấy lợi.
- **Bệnh đoán không khoanh phạm vi:** `QueryAnalysis` có thêm `disease_named`. Bệnh Groq chỉ suy từ
  cách gọi chung chung (ví dụ "bệnh đốm trên cây táo" → `black_rot`) nằm ở
  `ResolvedQuery.suspected_disease`, không gửi sang RAG; RAG tìm trong mọi tài liệu của cây. Router
  vẫn trả lời (`ACCEPT_QUERY`) khi có cây + bệnh đoán. Ảnh cho bệnh thì dùng bệnh của ảnh.
- **Groq lỗi không còn trả 500:** hết quota, lỗi mạng, JSON sai → câu gốc đi thẳng sang RAG với
  `rewrite_query=true` (RAG tự viết lại câu nối tiếp), chỉ kèm cây/bệnh từ ảnh của lượt này;
  `debug.normalizer_failed=true`.
- Prompt normalizer: quy tắc `normalized_query` nhắm vào việc tìm tài liệu (đủ dấu, bỏ teencode,
  giữ mọi chi tiết, không chèn tên bệnh đoán), thêm `disease_named` và ví dụ câu không dấu/teencode.
- Test: 84 pass, 1 skip như cũ (thêm `tests/test_normalizer.py`; sửa test đang kiểm tra câu mẫu).
- Còn mở (xem báo cáo): Groq khôi phục dấu sai ở tên cây/bệnh ("phan trang" → "phân trang"), và router
  không cho 21/58 câu hợp lệ đi tiếp. Nên thêm tên tiếng Việt có dấu của cây/bệnh vào prompt và nới router.

## 2026-09-23 — Từ chối ảnh không phải lá cây

- Ảnh gửi lên đi qua GenYOLO trước tiên. Không thấy lá thì `/api/chat` trả **HTTP 400**
  `"Ảnh không hợp lệ: mình không thấy lá cây nào trong ảnh…"` ngay, không chạy
  ConvNeXt/IEViT, không gọi Groq, không lưu lượt chat. File không mở được như ảnh
  cũng trả 400.
- Luật: hợp lệ khi có một lá với độ tin cậy ≥ `LEAF_MIN_CONFIDENCE` (0.6), hoặc lá
  (≥ 0.25) phủ ≥ `LEAF_MIN_AREA_RATIO` (25%) diện tích ảnh. Đo trên 24 ảnh PlantVillage
  và 26 ảnh không phải lá (COCO, ảnh chụp màn hình, ảnh trơn, nhiễu): đúng 50/50.
  **Chưa thử với ảnh chụp ngoài vườn**; nếu ảnh lá thật bị từ chối, hạ hai ngưỡng này.
- Pipeline: bước nhận diện ảnh chạy trước bước chuẩn hóa Groq. `last_detection` chỉ được
  lưu khi đã có câu trả lời, nên Groq/RAG lỗi không để lại hội thoại rỗng trong Mongo.
- `app.js`: lỗi 400/503 hiện đúng thông báo tiếng Việt của backend thay cho câu chung.

## 2026-09-23 — Giao diện: bản Garden 17/09 + admin dạng bảng

Hệ thiết kế khóa trong `design.md`, token dùng chung ở `app/static/tokens.css`.

- **Chat (`index.html`, `styles.css`, `app.js`, `tokens.css`):** chép nguyên bản redesign
  ngày 17/09 từ `plant_disease_chatbot` (theme Garden, side rail, thẻ "Kết quả gần đây").
  API không đổi nên chạy thẳng với backend đã refactor.
- **Admin (`admin_review.html`, `admin.css`, `admin.js`):** vẫn là bảng như cũ, dựng lại
  theo hệ Garden (font, màu, icon, nút). Thêm: tên cây · bệnh dưới câu hỏi; tài liệu RAG
  đã dùng (hoặc cảnh báo "kho chưa có tài liệu") dưới câu trả lời của bot; ghi chú khi
  `ANSWER_BACKEND=rag` chưa dùng ví dụ Feedback RAG; nháp câu trả lời đúng không mất khi
  đổi bộ lọc; thông báo nằm ở thanh dưới thay cho toast.
- **Backend:** thêm `GET /admin/status`. Sửa lỗi Feedback RAG luôn lưu cây rỗng
  (`admin.py` đọc `metadata.planttype`, pipeline lưu `metadata.plant`).
- Kiểm tra bằng Chromium headless ở 320–1440px: trang không tràn ngang, không lỗi JS;
  chạy thử lọc, chọn, chặn khi thiếu câu trả lời đúng, giữ nháp, xóa.

## 2026-09-23 — Tích hợp RAG-module theo `2026-09-23-detection-server-refactor-plan.md`

### Đã làm

| Mục | Trạng thái | Ghi chú |
| --- | --- | --- |
| 2.1 Package `app/plant_ai/models/` | Xong | Thêm `__init__.py`. `.gitignore` của module đổi `models/` → `/models/`. `.gitignore` ở gốc repo cũng có `models/` và `*/models/`, nên đã thêm ngoại lệ `!detection-server-module/app/plant_ai/models/`. Đã kiểm tra trên một repo tạm: package được track, còn `models/*.pt` vẫn bị bỏ qua. ConvNeXt và cả 9 checkpoint IEViT đều nạp được với `strict=True`. |
| 2.2 Tên checkpoint ớt trên HF | **Chưa** | Việc này phải do chủ repo HF làm. Ngày 23/09 file trên HF vẫn tên `best_ievit_stage3_pepper _yolo.pth` (có dấu cách). Code không sửa theo tên sai, đúng như plan. |
| 2.3 Cây không có IEViT | Xong | `predict_image` bỏ qua bước YOLO/IEViT và trả `disease=None`, độ tin cậy lấy theo cây. `DetectionResult.disease` giờ là optional. Lượt chat vẫn đi tiếp và được lưu. |
| 2.4 `requirements.txt` | Xong | Gộp đủ gói, ghim phiên bản. Trên venv mới: `pip install -r requirements.txt` → `pip check` sạch → `import app.api` chạy được → toàn bộ test pass. |
| 2.5 `.env.example` | Xong | Không chứa secret. `app/__init__.py` và `rag/config.py` nạp `.env` vào môi trường, nên các biến đọc bằng `os.getenv` (`YOLO_*`, `RAG_*`...) cũng lấy được từ `.env`. |
| 3.1 `RagAnswerRequest` | Xong | Nằm trong `app/schemas.py`, `extra="forbid"`. Payload được cắt theo giới hạn của hợp đồng. Không gửi `rewrite_query`. |
| 3.2 Nhãn chuẩn trong `retrieval_query` | Xong | Xem `app/chat/labels.py`. Test xác nhận cả 33 class IEViT đều ra đúng key của prompt normalizer. |
| 3.3 `AnswerBackend` | Xong | Nằm trong `app/answer/`. `GroqAnswerBackend` bọc đúng luồng cũ, còn `RagHttpBackend` gọi `POST /v1/answer`. Bước 7–8 của pipeline giờ chỉ còn một lời gọi `answer_backend.answer(...)`. |
| 3.4 Cấu hình | Xong | `ANSWER_BACKEND` (mặc định `groq` cả trong code lẫn `.env.example`), `RAG_API_URL`, `RAG_API_KEY`, `RAG_API_TIMEOUT`. Đổi sang `rag` chỉ cần sửa `.env`. |
| 3.5 Response/lỗi | Xong | `sources`, `scope_status`, `grounded` được lưu vào `metadata` của bản ghi feedback. Lỗi ánh xạ theo bảng trong plan (401/502/503/timeout → 503, 422 → 500). Khi lỗi thì không lưu lượt chat. |
| 3.6 Hợp đồng frontend | Xong | Giữ nguyên 5 trường. Thêm trường `sources`. |
| 3.7 Feedback RAG | Theo mặc định của plan | Với `groq`: như cũ. Với `rag`: vẫn thu feedback, nhưng không gửi ví dụ sang RAG. Không xóa dữ liệu. |
| 4 Test | Xong | Thư mục `tests/` chạy offline: import, payload trong 4 tình huống, lỗi không lưu lượt, ảnh cam/bí, `RagHttpBackend` với `MockTransport`, ánh xạ nhãn, các mục P2. Project gốc không có test để chép sang. |
| 5 P2 | Xong | CORS theo `CORS_ORIGINS`. Xác thực HTTP Basic cho `/admin` và `/admin/*` (`ADMIN_USERNAME`/`ADMIN_PASSWORD`; để trống thì mở như cũ và ghi cảnh báo lúc khởi động). `response_model` + tag cho mọi route. `lifespan` thay `on_event`. `print()` → `logging`, lịch sử và toàn văn tài liệu chỉ ở mức DEBUG. Lỗi 500 không lộ exception. `MAX_HISTORY_TURNS` giờ quyết định số message đưa vào LLM (mặc định 6 = như trước). `MODEL_DIR` tính theo thư mục module. |

### Phạm vi

Chỉ sửa `detection-server-module/` (và thêm ngoại lệ `models/` vào `.gitignore` gốc).
RAG-module không đổi: RAG server là việc của bên RAG, detection chỉ cần gửi đúng
payload theo `2026-09-23-rag-api-openapi.json`.

### Kiểm tra đã chạy

- Python 3.10.12, Linux, torch 2.14.0+cpu. **Chưa chạy thử với CUDA.**
- Lệnh test: `python -m pytest -q tests` → 76 passed.
- Venv mới: `pip install -r requirements.txt` → `pip check` sạch → `import app.api` chạy được.
- ConvNeXt và 9 checkpoint IEViT tải từ HF, nạp `strict=True` được.
- `RagHttpBackend` kiểm tra với `httpx.MockTransport` theo đúng file OpenAPI: header
  `Authorization`, các trường gửi đi, bảng ánh xạ lỗi 401/422/502/503/timeout.
- **Chưa chạy:** Groq + MongoDB thật (chưa có `GROQ_API_KEY`), và `ANSWER_BACKEND=rag`
  với RAG server thật (bên RAG chưa cung cấp server).
