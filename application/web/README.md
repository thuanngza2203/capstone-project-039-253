# Web — Bác sĩ cây trồng

Web cho người dùng (chat bằng chữ và ảnh) và trang quản trị, chạy trên máy tính và điện thoại.
React 18 + Vite 5, JavaScript. Thiết kế theo `RAG-module/agents/2026-09-24-system-architecture-api-frontend-plan.md`.

| Đường dẫn | Trang |
| --- | --- |
| `/` | Chat: gửi chữ, ảnh (tự thu nhỏ còn cạnh dài 1.600 px), hoặc cả hai. Nút chọn model ở đầu trang (kiểu ChatGPT / Gemini) khi detection cho phép nhiều model (`CHAT_LLM_PROVIDERS`). Nút "Tìm web" trong ô nhập: Groq tìm web và trả lời, không dùng kho tài liệu; câu trả lời kèm các trang web đã đọc và cảnh báo thông tin chưa được kiểm chứng. Mỗi câu trả lời liệt kê tài liệu trong kho và link nguồn tham khảo, nút thích/không thích, và mục "Cách hệ thống xử lý câu này" hiện từng bước pipeline (nhận diện ảnh, Groq chuẩn hóa, cây/bệnh, điều hướng, câu gửi RAG, phạm vi tìm). `/?session=<id>` mở một cuộc trò chuyện có sẵn. Trang chat không có đường dẫn sang trang quản trị. |
| `/admin/login` | Đăng nhập quản trị bằng `ADMIN_USERNAME` / `ADMIN_PASSWORD` trong `.env` của detection (phải đặt `ADMIN_PASSWORD`). Mọi trang `/admin/*` chuyển về đây khi chưa đăng nhập hoặc phiên đã hết hạn. |
| `/admin` | Tổng quan: số câu trả lời, cuộc trò chuyện, tỉ lệ có tài liệu, tỉ lệ được thích; biểu đồ theo ngày, theo hành động, bệnh hỏi nhiều, phạm vi tìm, đánh giá. |
| `/admin/conversations` | Mọi cuộc trò chuyện; từng lượt kèm dấu vết pipeline, mở tiếp trong trang chat. |
| `/admin/feedback` | Như trang admin cũ: lọc, tìm, sửa câu trả lời đúng, đưa vào Feedback RAG, xóa. |
| `/admin/kb` | Kho tri thức: index, tài liệu, chunk, cây · bệnh, tìm chunk theo nội dung. |
| `/admin/kb/playground` | Chạy riêng RAG (tìm, sinh câu trả lời) với mọi tùy chọn: cây/bệnh, index, cách tìm, số chunk, reranker, LLM. |
| `/admin/system` | Trạng thái detection, RAG, LLM (model thật đang chạy trên Vast). |

## Chạy local

Cần Node 18 trở lên (đã thử Node 22).

```powershell
cd application/web
npm install
npm run dev          # http://localhost:5173
```

Web gọi backend qua proxy của Vite: `/detection/*` → `http://127.0.0.1:8005`, `/rag/*` → `http://127.0.0.1:8010`
(đổi bằng `DETECTION_TARGET`, `RAG_TARGET` trong `.env.local` nếu backend chạy chỗ khác). Vì vậy phải chạy
trước, mỗi thứ một cửa sổ:

```powershell
# 1. SSH tunnel tới Vast (cổng 8000), theo README của LLM-server-module.
# 2. RAG server
cd RAG-module; python -m server                                           # :8010
# 3. detection-server
cd detection-server-module; uvicorn app.api:app --host 127.0.0.1 --port 8005
```

**Để chat đi hết pipeline (detection → RAG → model trên Vast)**, kiểm tra `.env` (đã chạy thử ngày 25/09 với
các giá trị này truyền qua biến môi trường):

| File | Biến | Giá trị cần có | Hiện tại |
| --- | --- | --- | --- |
| `detection-server-module/.env` | `ANSWER_BACKEND` | `rag` (không thì chat dùng `rag/` nội bộ + Groq, không qua RAG server) | `groq` |
| `detection-server-module/.env` | `MONGO_URI` | đúng MongoDB đang chạy | `…:27018`, trong khi `mongod` trên máy nghe `27017` |
| `RAG-module/.env` | `VLLM_MODEL` | `rag-llm` (served-model-name của vLLM) | `Qwen/Qwen3.8-27B-FP8` → `/v1/answer` lỗi 404 |

Trang **Hệ thống** (`/admin/system`) báo ngay mấy lỗi cấu hình này.

## Mở trên điện thoại (cùng Wi-Fi)

1. `npm run dev` in ra dòng `Network: http://192.168.x.x:5173`; mở địa chỉ đó trên điện thoại.
2. Lần đầu có thể phải mở cổng trong Windows Firewall (PowerShell quyền Administrator):
   `New-NetFirewallRule -DisplayName "Plant web" -Direction Inbound -Protocol TCP -LocalPort 5173,4173 -Action Allow -Profile Private`

Qua `http://<IP>` trình duyệt không có `crypto.randomUUID()`; web tự tạo `session_id` bằng
`crypto.getRandomValues()` nên vẫn chạy. Chọn ảnh bằng `<input type="file">` vẫn mở được camera.

## Build và deploy

```powershell
npm run build        # ra dist/
npm run preview      # http://localhost:4173, vẫn dùng proxy như dev
```

Khi đặt web và API ở địa chỉ public khác nhau, ghi `.env.production` (xem `.env.example`) rồi build lại:
`VITE_DETECTION_URL`, `VITE_RAG_URL`. Lúc đó detection cần `CORS_ORIGINS=*` trong `.env`; RAG đã mở CORS
mặc định (`RAG_API_CORS_ORIGINS`). Đặt `dist/` lên hosting tĩnh thì bật chuyển mọi đường dẫn về `index.html`.

## Cấu trúc

```
src/
  api.js                    mọi lời gọi detection và RAG
  lib/                      localStorage, thu nhỏ ảnh, định dạng số/ngày, tên cây/bệnh tiếng Việt
  components/               icon, markdown, biểu đồ, DebugPanel (dấu vết pipeline), chat/
  pages/Chat.jsx            trang người dùng
  pages/admin/              trang quản trị (tải riêng, người dùng chat không phải tải)
  styles/tokens.css         hệ màu Garden, chép từ detection-server-module/app/static
```

Danh sách cuộc trò chuyện ở trang chat lưu trong `localStorage` của từng trình duyệt; backend vẫn lưu nội dung
(MongoDB). Trang admin thấy mọi cuộc trò chuyện.
