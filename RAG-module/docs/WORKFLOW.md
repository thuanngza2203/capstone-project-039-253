# Workflow sử dụng hằng ngày

[← README](../README.md) · [Hướng dẫn dữ liệu](DATA_GUIDE.md) · [Pipeline](PIPELINE.md)

## Thêm hoặc sửa tài liệu

1. Đặt file UTF-8 tại `data/<crop>/<ten_tai_lieu>.txt`.
2. Kiểm tra file có nội dung và filename mô tả đúng chủ đề.

Mỗi file nên chỉ mô tả một bệnh. Ghi rõ tên chuẩn và các tên gọi khác gần đầu tài liệu để pipeline tạo `disease_id`, `disease_aliases` và identity header ổn định.
3. Chạy lại toàn bộ index:

   ```powershell
   python main.py index
   ```

4. Kiểm tra retrieval trước:

   ```powershell
   python main.py search "Bệnh ghẻ táo có dấu hiệu gì trên lá?"
   ```

5. Khi các chunk đúng đã nằm ở đầu kết quả, kiểm tra end-to-end:

   ```powershell
   python main.py ask "Bệnh ghẻ táo có dấu hiệu gì trên lá?"
   ```

Không chỉnh trực tiếp file bên trong `chroma_db/`; thư mục này luôn có thể tạo lại từ `data/`.

Sau khi source được nâng cấp để có disease identity, phải build lại index cũ. Khi smoke test, dùng cả một câu nêu rõ tên bệnh và một câu chỉ mô tả triệu chứng: câu đầu phải được filter đúng bệnh, câu sau phải được tìm trên toàn corpus.

## Debug khi câu trả lời sai hoặc thiếu

Chạy `search` bằng chính câu hỏi gây lỗi, rồi kiểm tra theo thứ tự:

- **Không thấy source đúng:** xác nhận file nằm trong `data/`, có đuôi `.txt`, đúng UTF-8 và đã chạy lại `index` sau lần sửa gần nhất.
- **Câu hỏi nêu rõ bệnh nhưng sang source khác:** kiểm tra `disease`, `disease_id`, `disease_aliases`; sửa tên/alias bị thiếu hoặc trùng rồi build lại index.
- **Câu hỏi chung tìm nhiều bệnh:** đây là hành vi dự kiến vì query không khớp duy nhất một `disease_id`; đánh giá thứ tự dense search thay vì chờ một filter.
- **Source đúng nhưng chunk thiếu phần quan trọng:** đặt thông tin liên quan gần nhau hơn trong TXT hoặc thử điều chỉnh chunk size/overlap.
- **Chunk đúng nhưng đứng quá thấp:** thử `--top-k 6` trên lệnh `search` trước, hoặc viết câu hỏi bằng thuật ngữ xuất hiện trong tài liệu.
- **Context đúng nhưng câu trả lời sai:** xem prompt và thử `GEMINI_MODEL`; đây là vấn đề generation, không phải indexing.
- **Nguồn trả về không nói về câu hỏi:** corpus có thể chưa có tài liệu phù hợp. Không nên ép LLM suy đoán ngoài dữ liệu.

Danh sách nguồn in sau câu trả lời là các file đã được retrieval chọn, không có nghĩa mọi câu trong file đều được Gemini sử dụng.

## Điều chỉnh retrieval có kiểm soát

Chỉ đổi một biến tại một thời điểm và giữ lại vài câu hỏi kiểm tra cố định.

1. Lưu 5–10 câu hỏi đại diện và source mong đợi.
2. Chạy `search`, ghi lại thứ tự kết quả hiện tại.
3. Chỉ thay một yếu tố:
   - `CHUNK_SIZE`: chunk lớn chứa nhiều ngữ cảnh hơn nhưng có thể pha nhiều chủ đề.
   - `CHUNK_OVERLAP`: overlap lớn giảm mất ý ở ranh giới nhưng tăng số chunk.
   - Với top-k, thử `python main.py search "<câu hỏi>" --top-k 6` trước khi đổi mặc định `TOP_K`.
4. Nếu đổi chunk size/overlap, chạy lại `index`. Nếu chỉ đổi `TOP_K`, không cần rebuild.
5. Chạy lại cùng bộ câu hỏi và chỉ giữ thay đổi nếu retrieval tốt hơn rõ ràng.

Đổi `EMBEDDING_MODEL` hoặc thiết lập normalize luôn yêu cầu build lại index. Đổi `GEMINI_MODEL` không yêu cầu build lại.

Thay đổi tên chuẩn, alias, `disease_id` hoặc cách tạo identity header cũng yêu cầu build lại index vì cả metadata và nội dung embedding đã đổi.

## Thêm một loại cây trồng mới

Ví dụ thêm tài liệu cà chua:

```text
data/
├── apple/
│   ├── apple_black_rot.txt
│   └── apple_scab.txt
└── tomato/
    ├── tomato_early_blight.txt
    └── tomato_late_blight.txt
```

Checklist trước khi coi dữ liệu mới là sẵn sàng:

- Mỗi file chỉ tập trung vào một bệnh hoặc một chủ đề rõ ràng.
- Nội dung là UTF-8, có tiêu đề và thuật ngữ người dùng có thể hỏi.
- Nguồn tham khảo và thời điểm cập nhật được ghi trong nội dung.
- Thông tin thuốc/hoạt chất không có liều lượng suy đoán; khuyến cáo tuân thủ nhãn và quy định địa phương.
- `python main.py index` hoàn tất và báo số tài liệu/chunk hợp lý.
- Một câu dùng tên bệnh chuẩn và một câu dùng alias đều tìm đúng file bằng `search`.
- Một câu mô tả triệu chứng chung và một câu so sánh nhiều bệnh được thử để xác nhận retrieval không filter nhầm.
- Ít nhất ba câu hỏi tiếng Việt tìm đúng file bằng `search`.
- Một câu ngoài phạm vi được thử bằng `ask` và hệ thống thừa nhận thiếu dữ liệu.
- Nếu cần tiếng Anh, thêm một smoke test tiếng Anh; V1 không đảm bảo chất lượng tương đương tiếng Việt.

## Trước khi commit

```powershell
python -m pytest
```

Không commit `.env`, `chroma_db/`, virtual environment hoặc cache model. Commit source, test, tài liệu và các file dữ liệu đã được phép chia sẻ.

Chi tiết định dạng TXT nằm trong [DATA_GUIDE.md](DATA_GUIDE.md); lỗi cài đặt thường gặp nằm trong [SETUP.md](SETUP.md).
