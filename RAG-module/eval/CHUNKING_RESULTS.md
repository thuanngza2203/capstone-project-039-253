# Kết quả chunking A/B — 17/09/2026

## Kết luận sử dụng

Đã triển khai cả `recursive` và `structure`, chọn bằng env. Giữ `recursive`
làm mặc định để người dùng thử đối chiếu. `structure` cải thiện điểm tổng thể
trên bộ hồi quy này, nhưng còn regression ở một số câu. Chưa đánh giá reranker
thật hoặc chất lượng câu trả lời do LLM sinh ra.

## Thiết lập

- A: 22 file snapshot trước migration, recursive 1000 ký tự/overlap 150,
  identity header cũ; 362 chunk.
- B: cùng nội dung chuyên môn, có marker heading, structure tối đa 400 token,
  overlap 40, compact header; 703 chunk.
- Embedding thật: `AITeamVN/Vietnamese_Embedding`, cosine.
- Retrieval hybrid; candidate-k 20, RRF-k 60, top-k 4; tắt reranker ở cả A/B.
- 60 câu cố định trong `chunking_queries.jsonl`: 40 dev, 20 heldout.
  Hai câu ngoài corpus chỉ là probe; tính điểm trên 58 câu có bằng chứng.
- Các câu, nhãn, budget 400 và cách chunking được chốt trước lần retrieval này.
  Nhãn do người triển khai soạn từ corpus; chưa được chuyên gia/người dùng duyệt.

## Chỉ số

| Chỉ số | Recursive A | Structure B |
| --- | ---: | ---: |
| Hit@1, toàn bộ câu có đáp án | 34/58 (58,6%) | 40/58 (69,0%) |
| Recall@4 evidence, toàn bộ | 84,5% | 89,7% |
| MRR@4, câu một bằng chứng | 0,705 | 0,776 |
| Hit@1, heldout có đáp án | 12/18 | 12/18 |
| Recall@4 evidence, heldout | 83,3% | 88,9% |
| Đủ required_context, 4 câu gán nhãn riêng | 4/4 | 4/4 |
| Context token trung vị, toàn bộ | 1.411 | 753,5 |
| Retrieval latency trung vị | 0,317 s | 0,383 s |
| Retrieval latency P95 | 0,407 s | 0,506 s |

Thời gian tạo A khoảng 27,06 s; B khoảng 13,33 s. A bao gồm khởi tạo model ở
lần đầu, B dùng lại model trong process, nên **không dùng hai số này để kết
luận B index nhanh gấp đôi**. Thời gian cũng phụ thuộc tải máy hiện tại.

So sánh theo `(Recall@4, Hit@1)`: 14 câu tốt hơn, 39 không đổi, 5 xấu hơn.
Recall là trung bình tỷ lệ các đoạn evidence được phủ, không phải tỷ lệ câu
trả lời đúng. Q058 có hai nguồn nên mỗi nguồn được tính riêng. Hợp các span
được dùng để không đếm trùng overlap, bỏ whitespace-only gap giữa các chunk.

## Regression cần giữ lại để sửa tiếp

| Query | A → B `(Recall@4, Hit@1)` | Quan sát top 1 của B |
| --- | --- | --- |
| q009: Phấn trắng trên lá anh đào non | `(1,0) → (0,0)` | Lấy mục triệu chứng trên quả |
| q040: Protein V1 trong TYLCV | `(1,1) → (0,0)` | Lấy lời dẫn cấu trúc bộ gen, chưa lấy body V1 trong top 4 |
| q047: Cắt ngang thân nho nghi Esca | `(1,1) → (1,0)` | Lấy mục kiểm tra nhanh; evidence chuẩn vẫn trong top 4 |
| q048: Đốm mới trên lá đào ướt hay khô | `(1,1) → (0,0)` | Lấy mục tóm tắt |
| q051: Lá bí trắng có chắc phấn trắng | `(1,1) → (1,0)` | Lấy mục kiểm tra nhanh; evidence chuẩn vẫn trong top 4 |

Nhãn trích một vị trí bằng chứng cụ thể. Một đoạn tóm tắt khác có thể chứa
thông tin tương đương nhưng chưa được gán nhãn; vì vậy cần đọc hit trước khi
kết luận mọi regression đều là sai kiến thức. Trường `wrong_section_hits`
trong JSON chỉ là proxy: cùng file nhưng không giao đoạn evidence/condition
đã gán nhãn; không phải mọi chunk đó đều vô ích.

## Kiểm tra token trên toàn corpus

| Strategy/budget | Chunk | Token median | P95 | Max | Header ratio ước tính |
| --- | ---: | ---: | ---: | ---: | ---: |
| Recursive | 362 | 360 | 397 | 429 | 25,74% |
| Structure 300 | 718 | 144 | 284 | 300 | 34,06% |
| Structure 400 | 703 | 143 | 285 | 399 | 33,75% |
| Structure 450 | 697 | 141 | 286 | 450 | 33,63% |

Đây là token từ tokenizer embedding, không phải tokenizer reranker. Budget
300/450 mới được kiểm tra preview/boundary, chưa chạy retrieval riêng; chưa
kết luận 400 là tối ưu. Header ratio đếm header riêng nên chỉ là ước tính.
Số chunk tăng do giữ riêng các mục trong tài liệu, kéo theo chi phí BM25 cao hơn.

## Bảo toàn nội dung và tương thích

- Inverse migration khôi phục nguyên văn cả 22 file, kể cả xuống dòng canonical.
- Metadata identity trước/sau giống nhau ở mọi file.
- Recursive trên snapshot raw sinh đúng 362 text và start offset như code cũ.
- Preview toàn corpus không có warning; 5 record metadata của apple scab không
  trở thành chunk bằng chứng.
- `python -m pytest -q`: **156 passed**, 6 warning dự kiến cho index legacy và
  fixture heading rỗng. Pytest kiểm tra hai strategy/index độc lập, sai embedding, build dang dở,
  unit span/coverage, Unicode, heading, token budget, preview không tải weights,
  các API/CLI cũ và phép tính benchmark.

## Artifacts và cách tái lập

Report chi tiết có đầy đủ text/metadata từng hit:
`artifacts/chunking/benchmark.json`.
Index A/B: `artifacts/chunking/recursive_db` và `artifacts/chunking/structure_db`.
Preview: `recursive.jsonl`, `structure-300.jsonl`, `structure-400.jsonl`,
`structure-450.jsonl` cùng các file `.summary.json`.

```powershell
python scripts/benchmark_chunking.py --build
```

Lần đầu bị sandbox chặn truy cập Hugging Face; lần chạy lại đã được cấp quyền
và dùng embedding thật. Sau khi chạy, cùng các hit được chấm lại bằng phép đo
bỏ whitespace-only gap; không đổi query, nhãn, thứ hạng hoặc index để tăng điểm.
Các chỉ số Hit/Recall/MRR không thay đổi sau sửa cách tính này.

## Giới hạn còn lại

- C/D có reranker và E giữ header cũ cho structure chưa chạy; không quy toàn bộ
  mức tăng cho boundary, cũng chưa xác nhận evidence không bị reranker cắt.
- Chưa kiểm tra câu trả lời Ollama/qwen3.5:4b trong benchmark này.
- Chưa có nhãn mức độ liên quan để tính nDCG hoặc kiểm tra khả năng từ chối với LLM.
- Heldout nhỏ; chỉ tăng tương đương một câu về Recall@4. Cần thêm câu người dùng
  thật và các biến thể nhiều chủ đề trước khi đổi mặc định.
