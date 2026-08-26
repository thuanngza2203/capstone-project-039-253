# Cài đặt và cách chạy

[← README](../README.md) · [Hướng dẫn dữ liệu](DATA_GUIDE.md) · [Workflow](WORKFLOW.md)

## Yêu cầu

- Python 3.11.
- Kết nối Internet ở lần cài dependency, lần tải embedding model đầu tiên và khi dùng Gemini.
- Gemini API key nếu dùng lệnh `ask`.
- Dung lượng trống cho môi trường Python, model Hugging Face và Chroma.

`index` và `search` chạy local sau khi embedding model đã được tải. `ask` luôn cần mạng trong V1.

## 1. Tạo môi trường Python

Mở PowerShell tại `RAG-module-2`:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu PowerShell chặn script kích hoạt, có thể dùng Python trực tiếp mà không đổi execution policy:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py index
```

## 2. Cấu hình `.env`

Tạo file từ mẫu:

```powershell
Copy-Item .env.example .env
```

Các giá trị hỗ trợ:

```dotenv
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=AITeamVN/Vietnamese_Embedding
EMBEDDING_DEVICE=cpu
```

Không commit `.env`. Key chỉ cần thiết cho `ask`; có thể để trống khi chỉ build hoặc kiểm tra retrieval.

## 3. Chuẩn bị dữ liệu

Dự án đã có template tham khảo tại `data/apple/`. Thêm dữ liệu theo dạng:

```text
data/<crop>/<document_name>.txt
```

File phải là UTF-8 và không rỗng. Xem template cùng quy tắc nội dung trong [DATA_GUIDE.md](DATA_GUIDE.md).

## 4. Build index

```powershell
python main.py index
```

Kết quả thành công sẽ báo số tài liệu và số chunk được lưu. Lần đầu, Hugging Face tải `AITeamVN/Vietnamese_Embedding`; thời gian phụ thuộc mạng và CPU. Những lần sau model thường được lấy từ cache.

Mỗi lần sửa dữ liệu, đổi embedding model, chunk size hoặc overlap, chạy lại lệnh này. Index được lưu tại `chroma_db/`.

Nếu nâng cấp từ bản chưa có identity header và metadata bệnh, bắt buộc chạy lại `python main.py index`; index cũ không tương thích với filter `disease_id`.

Lệnh `index` reset collection cũ trước khi ghi dữ liệu mới. Nếu tiến trình bị
ngắt, hết bộ nhớ hoặc lỗi giữa chừng, hãy chạy lại lệnh sau khi sửa nguyên nhân;
không chỉnh tay các file trong `chroma_db/`.

## 5. Kiểm tra retrieval

```powershell
python main.py search "Bệnh ghẻ táo có triệu chứng gì?"
python main.py search "Lá táo có đốm mắt ếch là bệnh gì?"
```

Output cho biết source và nội dung các chunk gần nhất. Nếu nguồn chưa đúng, sửa dữ liệu hoặc retrieval trước khi kiểm tra Gemini.

## 6. Hỏi Gemini

```powershell
python main.py ask "Cách quản lý bệnh thối đen trên táo?"
```

Output gồm câu trả lời và danh sách source đã được retrieval chọn. Chỉ top-4 chunk mặc định cùng câu hỏi được gửi tới Gemini.

## Chạy test

```powershell
python -m pytest
```

Test sử dụng fake embedding/chat model hoặc temporary Chroma; không tải model thật và không gọi Gemini.

## Dùng GPU NVIDIA (tùy chọn)

CPU là mặc định để setup dễ nhất. Chỉ chuyển sang CUDA khi driver và PyTorch CUDA đã hoạt động:

1. Cài bản PyTorch tương thích driver/CUDA theo trình chọn chính thức của PyTorch.
2. Kiểm tra:

   ```powershell
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. Đặt trong `.env`:

   ```dotenv
   EMBEDDING_DEVICE=cuda
   ```

4. Chạy lại `python main.py index` để xác nhận toàn bộ pipeline.

Không đặt `cuda` nếu lệnh kiểm tra trả `False`. Thiết bị chỉ ảnh hưởng tốc độ embedding, không thay Gemini.

## Troubleshooting

### Không tìm thấy Python 3.11

Kiểm tra `py -0p`. Nếu không có 3.11, cài Python 3.11 rồi tạo lại `.venv`.

### Lỗi tải model hoặc thiếu dung lượng

Kiểm tra Internet, proxy và dung lượng cache Hugging Face. Xóa cache chỉ khi bạn hiểu các model khác cũng có thể dùng chung cache; lần sau sẽ phải tải lại.

### `index` báo file không phải UTF-8

Mở đúng file được nêu trong lỗi và lưu lại với encoding UTF-8. Không dùng cách bỏ qua ký tự lỗi vì sẽ làm giảm chất lượng retrieval.

### `search` hoặc `ask` báo chưa có index

Chạy:

```powershell
python main.py index
```

Nếu đã chạy nhưng vẫn lỗi, xác nhận lệnh đang được gọi từ đúng thư mục `RAG-module-2`.

### `ask` báo thiếu API key

Kiểm tra file tên chính xác là `.env`, biến tên chính xác là `GEMINI_API_KEY`, sau đó mở terminal mới hoặc chạy lại lệnh. Không đặt key trong dấu ngoặc nhọn.

### Gemini báo quota, authentication hoặc network error

Đọc nguyên văn lỗi để phân biệt key sai, API chưa được cấp quyền, hết quota và lỗi mạng. Pipeline không giả lập câu trả lời khi Gemini thất bại. Trong lúc xử lý, vẫn có thể dùng `search` để kiểm tra phần local.

### Kết quả retrieval không đúng

Không bắt đầu bằng việc đổi model. Chạy checklist trong [WORKFLOW.md](WORKFLOW.md): xác nhận source, index mới nhất, nội dung chunk, rồi chỉ điều chỉnh một tham số mỗi lần.
