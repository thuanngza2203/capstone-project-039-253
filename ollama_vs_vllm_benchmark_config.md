# Cấu hình Benchmark Ollama vs vLLM

## Mục tiêu
So sánh hiệu năng serving/inference giữa Ollama và vLLM trên cùng phần cứng, cùng model family và cùng workload.

| Hạng mục | Ollama | vLLM |
|---|---|---|
| Model | `qwen3:1.7b-fp16` | `Qwen/Qwen3-1.7B` |
| Kiến trúc | Qwen3 1.7B | Qwen3 1.7B |
| Precision | FP16 | FP16 (`--dtype float16`) |
| GPU | RTX 3070 Ti 8 GB | RTX 3070 Ti 8 GB |
| API | `/v1/completions` | `/v1/completions` |
| Backend benchmark | `openai` | `openai` |
| Dataset | `random` | `random` |
| Input length | 512 tokens | 512 tokens |
| Output target | 128 tokens | 128 tokens |
| Random range ratio | `0` | `0` |
| Temperature | `0` | `0` |
| Thinking | Không áp dụng trực tiếp vì dùng raw completion | Không áp dụng trực tiếp vì dùng raw completion |
| Context length | 8192 | 8192 |
| Warmup requests | 3 | 3 |
| Benchmark requests | 20 | 20 |
| Concurrency | 1 | 1 |
| Request rate | `inf` | `inf` |
| Streaming | Có, để đo TTFT/ITL | Có, để đo TTFT/ITL |
| EOS | Cho phép dừng sớm | Cho phép dừng sớm |
| File kết quả | `ollama-c1.json` | `vllm-c1.json` |
| Metrics chính | TTFT, TPOT, ITL, E2E, throughput | TTFT, TPOT, ITL, E2E, throughput |

## Workload

```text
512 input tokens
        ↓
model xử lý prompt
        ↓
sinh tối đa ~128 output tokens
        ↓
đo:
- TTFT
- TPOT
- ITL
- E2E latency
- Output throughput
- Total token throughput
```

## Ý nghĩa metric

- **TTFT (Time To First Token):** thời gian từ khi gửi request đến khi nhận token đầu tiên. Thấp hơn là tốt hơn.
- **TPOT (Time Per Output Token):** thời gian trung bình cho mỗi output token sau token đầu tiên. Thấp hơn là tốt hơn.
- **ITL (Inter-Token Latency):** độ trễ giữa các token liên tiếp. Thấp hơn là tốt hơn.
- **E2E Latency:** thời gian từ lúc gửi request đến khi nhận xong toàn bộ response. Thấp hơn là tốt hơn.
- **Output Throughput:** số output token server sinh được mỗi giây. Cao hơn là tốt hơn.
- **Total Token Throughput:** tổng tốc độ xử lý input + output token. Cao hơn là tốt hơn.

## Lưu ý

- Hai engine không chạy model cùng lúc; benchmark tuần tự để không tranh VRAM.
- Ollama và vLLM phải dùng cùng context length là `8192` cho bài test này.
- Bài test dùng `/v1/completions`, vì vậy không benchmark chat template hoặc thinking mode.
- Nên benchmark thêm concurrency `2`, `4`, `8` sau khi hoàn tất bài test concurrency `1`.
