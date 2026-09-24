# Thí nghiệm so sánh model và chunking

Quy trình: **bộ câu hỏi → RAG API → câu trả lời của từng variant → Google Form ẩn danh → phân tích**.
Một *variant* là một cấu hình cố định: index (cách chunk) + model LLM + tham số retrieval.

Kết quả nằm trong `experiments/runs/` và `experiments/surveys/`, đã gitignore vì repo public.

## 1. Chuẩn bị

```bash
# Hai index phải build từ cùng data/ hiện tại (mỗi khi sửa data/ thì chạy lại).
python main.py index --strategy recursive
python main.py index --strategy structure

# Chạy RAG API (để HF_HUB_OFFLINE=1 nếu embedding đã tải về, tránh lỗi mạng tới Hugging Face).
python -m server
```

Kiểm tra trên Swagger (`http://127.0.0.1:8010/docs`) hoặc bằng curl:

- `GET /v1/status`: cả hai mục trong `indexes` phải có `matches_data: true`.
- `GET /v1/llm?provider=vllm&probe=true`: `reachable: true`, `detail: null`, và `served[0].root`
  đúng model đang host. LLM-server giữ tên API `rag-llm` khi đổi model, nên `root` là cách
  duy nhất để biết model thật. `VLLM_MODEL` trong `.env` của RAG phải bằng `served[0].id`.

## 2. Bộ câu hỏi

JSONL, mỗi dòng một câu. Xem `questions.example.jsonl`.

| Trường | Bắt buộc | Ý nghĩa |
| --- | --- | --- |
| `id` | có | Duy nhất; dùng để ghép câu trả lời giữa các variant |
| `question` | có | Câu người dùng hỏi |
| `plant_type`, `disease` | không | Giả lập nhãn detection gửi sang (lọc tài liệu theo cây/bệnh) |
| `retrieval_query`, `subject_context` | không | Như payload của detection |
| trường khác (`topic`, `note`...) | không | Chỉ để ghi chú, không gửi sang API |

Nên có đủ loại câu (triệu chứng, nguyên nhân, xử lý, phòng ngừa), đủ cây, vài câu gõ không dấu,
1–2 câu ngoài kho tài liệu. Không chép câu trong `eval/chunking_queries.jsonl` (đã dùng để
chỉnh chunking).

## 3. Sinh câu trả lời

```bash
python -m experiments.generate --questions experiments/questions.jsonl \
    --variant qwen4b-structure --index structure --llm-provider vllm
```

- Đứt tunnel giữa chừng thì chạy lại **đúng lệnh đó**: câu đã xong được giữ, chỉ chạy câu còn thiếu hoặc lỗi.
- Công cụ dừng nếu model thật, index, `top_k`/`mode`/`rerank` khác lần chạy trước của variant đó.
  Đổi cấu hình thì đặt `--variant` mới.
- Công cụ từ chối index không khớp `data/` (thêm `--allow-stale` nếu cố ý).
- `--limit 2` để thử nhanh trước khi chạy cả bộ.

### So sánh chunking (cùng model)

```bash
python -m experiments.generate --questions experiments/questions.jsonl --variant qwen4b-structure --index structure
python -m experiments.generate --questions experiments/questions.jsonl --variant qwen4b-recursive --index recursive
```

Thước đo chính của chunking là retrieval (`scripts/benchmark_chunking.py`, có nhãn bằng chứng).
Khảo sát người dùng chỉ là thước đo phụ.

### So sánh model (Vast chỉ host một model mỗi lúc)

1. Host model A trên Vast, kiểm tra `GET /v1/llm?probe=true`.
2. `python -m experiments.generate ... --variant modelA-structure --index structure --llm-provider vllm`
3. Đổi `LLM_MODEL_ID` bên LLM-server, khởi động lại, probe lại cho chắc `root` đã đổi.
4. `python -m experiments.generate ... --variant modelB-structure --index structure --llm-provider vllm`

Dùng cùng `VLLM_MAX_TOKENS` cho mọi model và đặt đủ lớn: câu bị cắt (`BỊ CẮT` trong log) bị loại
khỏi khảo sát. Muốn có mốc tham chiếu thì thêm một variant `--llm-provider gemini`.

## 4. Số liệu tự động

```bash
python -m experiments.stats experiments/runs/modelA-structure.jsonl experiments/runs/modelB-structure.jsonl
```

Bảng gồm: số câu lỗi, từ chối, bị cắt, trích dẫn sai nguồn, thời gian sinh (trung vị, P95),
token ra trung bình, độ dài câu trả lời.

## 5. Dựng Google Form

```bash
python -m experiments.survey build --a experiments/runs/modelA-structure.jsonl \
    --b experiments/runs/modelB-structure.jsonl --out experiments/surveys/model-ab \
    --title "Khảo sát câu trả lời về bệnh cây trồng"
```

- Mỗi cặp một trang, vị trí "Câu trả lời 1/2" đảo ngẫu nhiên có cân bằng (seed cố định, `--seed`).
- Bỏ markdown và `[Nguồn n]` (giữ bằng `--keep-citations`). Bỏ cặp lỗi, cặp giống hệt nhau,
  cặp có câu bị cắt (giữ bằng `--include-truncated`).
- Tối đa 12 cặp mỗi form (`--per-form`); nhiều hơn thì chia thành nhiều form đều nhau.

Đầu ra:

| File | Dùng để |
| --- | --- |
| `preview.md` | Đọc lại toàn bộ form trước khi tạo |
| `form.gs` | Vào script.google.com → Dự án mới → dán → chọn `createForms` → Chạy. Link form nằm ở Nhật ký thực thi |
| `key.json` | Khóa giải mã variant nào ở vị trí nào. **Không gửi kèm form** |

Script chưa được chạy thử trên tài khoản Google thật. Nếu Forms báo lỗi (ví dụ câu trả lời quá
dài cho phần mô tả), xem nhật ký lỗi rồi chỉnh.

## 6. Phân tích

Trong Google Form: **Phản hồi → ⋮ → Tải phản hồi xuống (.csv)**, rồi:

```bash
python -m experiments.survey analyze --key experiments/surveys/model-ab/key.json \
    --responses "Khảo sát (1_2).csv" "Khảo sát (2_2).csv" --out experiments/surveys/model-ab/report.md
```

Báo cáo gồm: tỉ lệ thắng khi bỏ hòa, khoảng tin cậy Wilson 95%, kiểm định dấu, kết quả theo đa số
từng câu và theo nhóm người trả lời.

**Cỡ mẫu.** Số lượt đánh giá không hòa cần để phát hiện chênh lệch (α = 0,05, power 0,8): 70/30 cần
khoảng 47, 65/35 khoảng 85, 60/40 khoảng 194. Các lượt của cùng một người hay cùng một câu không
độc lập, nên xem thêm kết quả theo câu.
