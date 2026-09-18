# Hướng dẫn chuẩn bị dữ liệu

[← README](../README.md) · [Cài đặt](SETUP.md) · [Workflow](WORKFLOW.md)

## Cấu trúc thư mục

V1 chỉ đọc đệ quy file `.txt` bên dưới `data/`:

```text
data/
├── apple/
│   ├── apple_black_rot.txt
│   └── apple_scab.txt
├── tomato/
│   └── tomato_late_blight.txt
└── potato/
    └── potato_early_blight.txt
```

Quy ước khuyến nghị:

- Thư mục cấp đầu tiên là tên cây trồng, viết thường và ổn định, ví dụ `apple`, `tomato`.
- Mỗi file chỉ mô tả một bệnh; đây là ranh giới để mô tả bệnh và truy vết nguồn.
- Filename dùng chữ cái Latin, chữ số và dấu gạch dưới; tên phải gợi đúng nội dung.
- File dùng UTF-8 và không được rỗng.

Loader/recursive vẫn đọc được plain text. Với `CHUNKING_STRATEGY=structure`, cần
một H1 `# Tiêu đề`, các mục H2 `##`, mục con H3 `###`, không nhảy cấp. Corpus hiện
tại đã được thêm marker; không bắt buộc các file có cùng danh sách mục.
Định dạng và ví dụ hiện hành: [CHUNKING.md](CHUNKING.md#4-định-dạng-dữ-liệu).

## Template plain text cho recursive

Hai file trong `data/apple/` là ví dụ đầu tiên. Với dữ liệu mới, có thể dùng mẫu thực hành sau và bỏ các mục không có thông tin đáng tin cậy:

```text
TÊN TÀI LIỆU
Bệnh <tên bệnh tiếng Việt> (<tên tiếng Anh>)

CÂY TRỒNG
<tên cây tiếng Việt và tiếng Anh nếu có>

TÊN BỆNH
Tên tiếng Việt: ...
Tên tiếng Anh: ...
Tên gọi khác: ...

TÁC NHÂN GÂY BỆNH
...

TRIỆU CHỨNG
Triệu chứng trên lá: ...
Triệu chứng trên quả: ...
Triệu chứng trên thân/cành/rễ: ...

ĐIỀU KIỆN PHÁT SINH VÀ LÂY LAN
...

PHÂN BIỆT VỚI BỆNH KHÁC
...

QUẢN LÝ VÀ PHÒNG NGỪA
...

TỪ KHÓA
...

NGUỒN THAM KHẢO
Tên tổ chức hoặc tác giả, tiêu đề, URL, ngày truy cập/cập nhật nếu có.
```

Không cần thêm JSON, YAML hoặc tự viết identity header. Pipeline dùng tên bệnh/
tên gọi và định danh file để tạo metadata. Recursive prepend header `Tài liệu` /
`Bệnh` / `Tên gọi`; structure dùng header gọn có đường dẫn mục.

Tên chuẩn và alias phải rõ ràng. Tên gọi chung có thể xuất hiện ở nhiều cây; ghi rõ cây trong tài liệu để hỗ trợ xếp hạng. Không dùng ký tự `|` bên trong một alias vì `disease_aliases` dùng ` | ` làm dấu phân cách nội bộ.

### Pipeline suy ra danh tính bệnh

1. Ưu tiên tên nằm trong mục `TÊN BỆNH` hoặc dòng có nhãn `Tên bệnh`, `Tên tiếng Việt`.
2. Thu thập alias từ tên tài liệu, phần trong ngoặc, `Tên tiếng Anh`, `Tên tiếng Việt` và `Tên gọi khác`; các alias trùng sau chuẩn hóa được loại bỏ.
3. Nếu template thiếu tên bệnh rõ ràng, dùng tiêu đề tài liệu hoặc filename làm fallback.
4. Tạo `disease_id` từ thư mục cây trồng và tên bệnh đã bỏ dấu/chuyển về dạng chuẩn, ví dụ `apple-ghe-tao`.

Chỉ các nhãn danh tính rõ ràng được dùng làm alias. Tên tác nhân, từ khóa và tên bệnh viết như nội dung thông thường trong phần so sánh không được tự động coi là alias. Không đặt các nhãn Tên tiếng Anh, Tên tiếng Việt hoặc Tên gọi khác cho bệnh phụ ở phần so sánh; mỗi file chỉ nên khai báo các nhãn này cho bệnh chính.

## Viết nội dung để retrieval tốt

- Viết câu đầy đủ, nêu rõ chủ thể thay vì dùng quá nhiều từ như “nó”, “bệnh này”.
- Đặt tiêu đề ngay trước phần nội dung liên quan.
- Giữ các thông tin thường được hỏi cùng nhau, ví dụ tên bệnh, triệu chứng và bộ phận bị hại.
- Ghi cả tên tiếng Việt, tiếng Anh, tên khoa học và tên gọi phổ biến nếu đã được nguồn xác nhận.
- Ghi tên bệnh và tên gọi khác ở các dòng có nhãn rõ ràng; không coi tên tác nhân, từ khóa hoặc bệnh được nhắc trong phần phân biệt là alias.
- Không nhồi nhiều bệnh không liên quan vào một file rất dài.
- Không lặp cùng một đoạn vào nhiều file nếu không cần thiết; duplicate có thể chiếm nhiều vị trí top-k.
- Ghi nguồn tham khảo ngay trong tài liệu để có thể kiểm tra lại kiến thức.

## An toàn với thuốc bảo vệ thực vật

Thông tin nông nghiệp có thể thay đổi theo quốc gia, cây trồng, mùa vụ và nhãn đăng ký. Khi viết phần quản lý:

- Chỉ ghi hoạt chất, sản phẩm, liều lượng hoặc thời gian cách ly khi có nguồn chính thức phù hợp với phạm vi dự án.
- Không suy ra liều dùng từ cây trồng hay bệnh khác.
- Ghi rõ khu vực và ngày cập nhật nếu quy định có thể thay đổi.
- Ưu tiên biện pháp quản lý tổng hợp, vệ sinh đồng ruộng và theo dõi điều kiện phát bệnh.
- Nhắc người dùng tuân thủ nhãn sản phẩm và quy định địa phương; tài liệu RAG không thay thế tư vấn chuyên môn tại hiện trường.

Prompt của V1 yêu cầu Gemini không tự tạo thuốc hoặc liều lượng, nhưng chất lượng và độ an toàn vẫn phụ thuộc dữ liệu đầu vào.

## Metadata và citation

Với file:

```text
data/apple/apple_scab.txt
```

loader giữ metadata nguồn sau và bổ sung danh tính bệnh:

```python
{
    "source": "apple/apple_scab.txt",
    "crop": "apple",
    "title": "Apple Scab",
    "disease": "Bệnh ghẻ táo",
    "disease_id": "apple-ghe-tao",
    "disease_aliases": "Bệnh ghẻ táo | Apple Scab | bệnh sẹo táo",
}
```

Ý nghĩa các trường danh tính:

| Trường | Ví dụ | Ý nghĩa |
|---|---|---|
| `disease` | `Bệnh ghẻ táo` | Tên chuẩn dùng để hiển thị |
| `disease_id` | `apple-ghe-tao` | ID mô tả bệnh của tài liệu, không tự dùng để filter query |
| `disease_aliases` | `bệnh ghẻ táo \| apple scab \| bệnh sẹo táo` | Tên chuẩn/tên gọi khác trong identity header hỗ trợ tìm kiếm |

`source` là đường dẫn tương đối từ `data/`, được dùng trong nhãn `[Nguồn n: apple/apple_scab.txt]` và danh sách source của `ask()`. Đổi tên, di chuyển file hoặc sửa danh tính bệnh làm thay đổi index, vì vậy cần chạy lại `python main.py index`.

Identity header là dữ liệu do code sinh trước embedding, không phải nội dung cần chép vào TXT. Nó giúp mọi chunk giữ tên tài liệu/bệnh/alias ngay cả khi đoạn gốc chỉ có triệu chứng hoặc biện pháp quản lý.

## Kiểm tra dữ liệu mới

Sau khi thêm file:

```powershell
python main.py index
python main.py search "<câu hỏi có đáp án trong file mới>"
python main.py ask "<câu hỏi có đáp án trong file mới>"
```

Nên thử thêm:

- Một câu dùng tên bệnh chính xác.
- Một câu dùng alias của cùng bệnh; kiểm tra nguồn mong đợi có được xếp cao không.
- Một câu chỉ mô tả triệu chứng.
- Một câu so sánh hai bệnh; kiểm tra có đủ nguồn của cả hai bệnh trong top-k.
- Một câu ngoài corpus để quan sát Gemini có thừa nhận context không đủ hay không.

V1 chưa có relevance threshold/evidence gate cứng, nên phép thử ngoài corpus là
smoke test cho prompt chứ không phải bảo đảm rằng mọi câu lạ đều bị từ chối.

## Định dạng chưa hỗ trợ

V1 không đọc PDF, Markdown, Word, ảnh hoặc tài liệu scan và không có OCR. Hãy tự trích xuất, kiểm tra và lưu nội dung thành `.txt` UTF-8 trước khi index. Không chỉ đổi phần mở rộng của file sang `.txt`; nội dung phải thật sự là plain text.
