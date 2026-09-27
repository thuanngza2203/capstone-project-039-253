# Web

Giao diện cho người dùng và trang quản trị, chạy được trên máy tính và điện thoại. React 18 + Vite 5.

| Đường dẫn | Trang |
|---|---|
| `/` | Chat bằng chữ và ảnh; chọn model trả lời; nút "Tìm trên web"; mỗi câu trả lời kèm tài liệu nguồn, nút thích/không thích và các bước hệ thống đã xử lý |
| `/admin/login` | Đăng nhập quản trị (`ADMIN_USERNAME` / `ADMIN_PASSWORD` trong `.env` của detection) |
| `/admin` | Thống kê: số câu trả lời, tỉ lệ được thích, bệnh hỏi nhiều... |
| `/admin/conversations` | Mọi cuộc trò chuyện, kèm các bước xử lý từng lượt |
| `/admin/feedback` | Duyệt đánh giá, sửa câu trả lời, đưa vào Feedback RAG |
| `/admin/kb`, `/admin/kb/playground` | Xem tài liệu, chunk; chạy thử tìm kiếm và sinh câu trả lời |
| `/admin/system` | Trạng thái detection, RAG, LLM |

## Chạy

Cần Node.js 18 trở lên (đã chạy thử Node 20, 22), và [detection server](../../detection-server-module/README.md) (cổng 8005), [RAG module](../../RAG-module/README.md) (cổng 8010) đang chạy.

```powershell
cd application/web
npm ci
npm run dev
```

Mở <http://localhost:5173>. Vite chuyển tiếp `/detection/*` tới `http://127.0.0.1:8005` và `/rag/*` tới `http://127.0.0.1:8010`. Backend chạy ở địa chỉ khác thì tạo `.env.local`:

```dotenv
DETECTION_TARGET=http://127.0.0.1:8005
RAG_TARGET=http://127.0.0.1:8010
```

## Mở trên điện thoại (cùng Wi-Fi)

`npm run dev` in ra dòng `Network: http://192.168.x.x:5173`; mở địa chỉ đó trên điện thoại. Nếu không vào được, mở cổng trong Windows Firewall (PowerShell quyền Administrator):

```powershell
New-NetFirewallRule -DisplayName "Plant web" -Direction Inbound -Protocol TCP -LocalPort 5173 -Action Allow -Profile Private
```

## Build

```powershell
npm run build     # ra thư mục dist/
npm run preview   # xem bản build ở http://localhost:4173
```

Khi web và API nằm ở các địa chỉ public khác nhau, tạo `.env.production` với `VITE_DETECTION_URL`, `VITE_RAG_URL` (xem `.env.example`) rồi build lại; detection cần thêm địa chỉ web vào `CORS_ORIGINS`. Khi chạy bằng Docker, web được build sẵn và phục vụ bằng nginx ([docker/nginx.conf](docker/nginx.conf)).

## Cấu trúc

```text
src/
  api.js           Mọi lời gọi tới detection và RAG
  pages/Chat.jsx   Trang chat
  pages/admin/     Các trang quản trị
  components/      Khung chat, markdown, biểu đồ, bảng các bước xử lý
  lib/             Lưu localStorage, thu nhỏ ảnh, tên cây/bệnh tiếng Việt
```

Danh sách cuộc trò chuyện ở trang chat lưu trong trình duyệt; nội dung hội thoại lưu ở MongoDB phía detection.
