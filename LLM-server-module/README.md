# LLM server trên Vast.ai

Module này chạy **riêng LLM bằng vLLM trên GPU Vast.ai**. `RAG-module` tiếp tục
chạy trên host của bạn và gọi API của model. Không cài dependencies RAG, tạo
embedding hay index tài liệu trên Vast.

```text
Host của bạn                              Vast.ai / Linux
RAG-module                                LLM-server-module
  câu hỏi + lịch sử
  → retrieval từ Chroma
  → câu hỏi + các chunk ─── API ─────────→ vLLM → model chọn trong .env
  ← câu trả lời có trích nguồn ──────────┘
```

RAG đã có client phù hợp: chọn `LLM_PROVIDER=vllm`, đặt URL/model/key của Vast.
`ollama` và `gemini` vẫn là các lựa chọn hiện có ở host.

## Các file

| File | Công dụng |
|---|---|
| `install.sh` | Tạo venv Linux riêng, cài vLLM và kiểm tra CUDA |
| `.env.example` | Cấu hình **server**: model, port, API key, VRAM/context |
| `serve.py` | Đọc `.env`, kiểm tra cấu hình rồi khởi động API có sẵn của vLLM |
| `check_api.py` | Kiểm tra model ID, xác thực và một câu trả lời, không chạy RAG |
| `rag-client.env.example` | Các dòng cần ghép vào `.env` của **RAG trên host** |
| `nginx.conf.example` | Proxy HTTPS tùy chọn khi muốn public API |
| `tests/` | Test offline, không tải model hoặc dùng GPU |

**Thứ tự:** [cài trên Vast](#1-cài-trên-vastai) → [chạy LLM](#2-chạy-llm-trên-vast)
→ [kết nối từ host](#3-kết-nối-rag-từ-host-bằng-ssh-tunnel).
Nếu cần URL public không giữ tunnel, xem [mục 4](#4-public-api-qua-https-tùy-chọn).

## 1. Cài trên Vast.ai

Các khối `bash` chạy trong **terminal Linux sau khi SSH vào Vast**. Các khối
`powershell` chạy trên **Windows của bạn**.

Hướng dẫn dùng instance Ubuntu 24.04 x86_64, Python 3.12, GPU NVIDIA và quyền
root/sudo. Chọn GPU có đủ VRAM cho model/context; mặc định script dùng một GPU,
context 8192, tối đa một sequence đồng thời và 90% VRAM cho vLLM khi GPU dành
riêng cho LLM. Đây là cấu hình khởi đầu, không phải cam kết model sẽ vừa mọi
GPU; giảm tỷ lệ nếu GPU còn chạy tiến trình khác.

File mẫu hiện chọn **Qwen3.8-27B-FP8 cho GPU 48 GiB**. Giữ hậu tố `-FP8`;
bản không có hậu tố này có yêu cầu bộ nhớ khác. Tổng context 8192 bao gồm câu
hỏi, lịch sử, tài liệu truy xuất và token trả lời.

Thêm SSH public key vào Vast và dùng lệnh Connect/SSH của instance để đăng
nhập. Instance dạng container không cần chạy Docker lồng bên trong.
[SSH Vast.ai](https://docs.vast.ai/guides/instances/connect/ssh),
[quản lý instance](https://docs.vast.ai/guides/instances/manage-instances)

### 1.1. Công cụ hệ thống

Dùng root cho hai lệnh apt, hoặc thêm `sudo` nếu đăng nhập user thường:

```bash
apt-get update
apt-get install -y git ca-certificates python3 python3-venv python3-pip tmux nano build-essential
nvidia-smi
python3 --version
```

Nếu `nvidia-smi` không thấy GPU, kiểm tra instance/template trước khi cài.
Không copy venv Windows sang Linux.

Vast thường mở sẵn tmux khi SSH. Dùng phiên đó; nếu chưa ở tmux thì chạy
`tmux new -s llm`. `Ctrl+B`, rồi `C` mở cửa sổ mới; `Ctrl+B`, rồi `D` detach
và giữ tiến trình. Không tạo tmux lồng nhau.

### 1.2. Chỉ lấy thư mục LLM server từ GitHub

Sau khi commit/push module mới từ host, trên Vast:

```bash
mkdir -p /workspace
cd /workspace
git clone --filter=blob:none --sparse https://github.com/thuanngza2203/capstone-project-039-253.git
cd capstone-project-039-253
git sparse-checkout set LLM-server-module
cd LLM-server-module
bash install.sh
source .venv/bin/activate
```

Nếu đã clone đầy đủ repository, chỉ cần `cd` vào `LLM-server-module` rồi cài;
không cần sparse checkout. Nếu code nằm trên branch khác, `git switch TEN_BRANCH`
trước khi cài. Repository private cần credential GitHub có quyền đọc; SSH key
đăng nhập Vast không tự cấp quyền đọc repository.

Các lệnh dưới dùng `/workspace/capstone-project-039-253`. Nếu bạn clone với tên
thư mục khác, thay đường dẫn đó bằng thư mục thực tế; dùng `pwd` để kiểm tra.

`install.sh` cài vLLM trong `.venv` của module này, dùng
`uv pip install ... --torch-backend=auto` và lưu phiên bản đã cài vào
`runtime/requirements.freeze.txt`. Requirements yêu cầu vLLM từ **0.17.0**,
phiên bản đã hỗ trợ Qwen3.5. Bản mới hơn phù hợp GPU/driver sẽ được chọn khi
cài lần đầu; không cần tự dùng nightly.
[Release vLLM 0.17.0](https://github.com/vllm-project/vllm/releases/tag/v0.17.0),
[Cài đặt vLLM](https://docs.vllm.ai/en/stable/getting_started/quickstart/)

Installer và launcher đều nhắm đến venv của Python đã chọn, tránh dùng nhầm
vLLM có sẵn trong template Vast. Nếu máy có nhiều Python, chọn interpreter
khi tạo venv mới, ví dụ `LLM_PYTHON_BIN=python3.12 bash install.sh`.

**Cài lần đầu / bổ sung dependency thiếu:** `bash install.sh`.
**Chủ động nâng cấp:** dừng LLM, sao lưu `runtime/requirements.freeze.txt`, rồi:

```bash
bash install.sh --upgrade
```

Chạy installer bình thường không ép nâng cấp toàn bộ stack đã cài, nhưng vẫn
cập nhật package nếu cần để thỏa requirements mới. Không cần chạy installer
mỗi lần mở server. Giữ bản freeze sau khi kiểm tra inference thực tế thành công.

### 1.3. Tạo cấu hình server

```bash
if [ ! -f .env ]; then
  cp .env.example .env
fi
python -c "import secrets; print(secrets.token_urlsafe(32))"
nano .env
```

Copy key vừa tạo vào `LLM_API_KEY` và giữ lại để điền vào `.env` của RAG.
Không tạo lại key ở mỗi lần khởi động. `.env` và model cache được Git ignore.
Trong nano: `Ctrl+O`, Enter để lưu; `Ctrl+X` để thoát.

Các giá trị chính:

```dotenv
LLM_MODEL_ID=Qwen/Qwen3.8-27B-FP8
LLM_SERVED_MODEL_NAME=rag-llm
LLM_PORT=8000
LLM_API_KEY=DIEN_KEY_BAN_VUA_TAO
LLM_MAX_MODEL_LEN=8192
LLM_MAX_NUM_SEQS=1
LLM_TENSOR_PARALLEL_SIZE=1
LLM_GPU_MEMORY_UTILIZATION=0.90
LLM_REASONING_PARSER=qwen3
LLM_LANGUAGE_MODEL_ONLY=true
LLM_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'
LLM_QUANTIZATION=
```

`LLM_MODEL_ID` là model tải từ Hugging Face, hoặc thư mục model đã tải trên
server. `LLM_SERVED_MODEL_NAME` là tên RAG gửi trong API request. Giữ tên API
`rag-llm` khi thay weights để phía RAG không phải đổi tên model theo.
File mẫu dùng [Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8),
tắt thinking mặc định ở server bằng `LLM_CHAT_TEMPLATE_KWARGS`; vLLM tự đọc
định dạng FP8 từ checkpoint khi `LLM_QUANTIZATION` trống. Theo
[recipe Qwen3.8](https://recipes.vllm.ai/Qwen/Qwen3.8-27B), runtime cần
`transformers>=5.8.0`; kiểm tra stack trong venv nếu đã cài từ trước.

**Nâng cấu hình cũ từ 4B lên 27B:** sửa `LLM_MODEL_ID` và các biến GPU/parser
như trên trong `.env` của Vast. Giữ API key và alias đang khớp với RAG, dừng
server cũ rồi chạy lại theo mục 2. Lấy code mới không cập nhật `.env` có sẵn.

## 2. Chạy LLM trên Vast

Trong cửa sổ tmux dành cho server:

```bash
cd /workspace/capstone-project-039-253/LLM-server-module
source .venv/bin/activate
python serve.py --dry-run
python serve.py
```

`--dry-run` kiểm tra cấu hình và in lệnh không có secret; không nạp model và
không kiểm tra được khả năng GPU thực thi model. Lần đầu `serve.py` tải weights
vào `models/huggingface/`; những lần sau dùng lại cache. Server vẫn phải nạp
weights vào GPU mỗi lần khởi động tiến trình.

Để nguyên cửa sổ server, mở cửa sổ tmux khác và đợi server sẵn sàng rồi chạy:

```bash
cd /workspace/capstone-project-039-253/LLM-server-module
source .venv/bin/activate
python check_api.py
```

Kết quả thành công phải có `API hoạt động, model: rag-llm` và câu trả lời.
Lệnh này gửi GET `/v1/models` rồi POST `/v1/chat/completions` có Bearer key.
Đây là kiểm tra kết nối/giao thức, không đánh giá chất lượng trả lời bệnh cây.
File `.env` phải tồn tại; `--env-file` sai đường dẫn sẽ báo lỗi trước khi gọi
API. File UTF-8 có BOM từ editor Windows cũng đọc được. Checker không theo
redirect để tránh chuyển key sang endpoint khác; cấu hình URL đích trực tiếp.

Checker dùng `LLM_CHECK_MAX_TOKENS` (mặc định 800); nếu đọc `.env` của RAG thì
dùng `VLLM_MAX_TOKENS`, `VLLM_TIMEOUT` và `VLLM_THINK` tương ứng. Có thể ghi đè
bằng `python check_api.py --max-tokens 2048 --timeout 180`. Checker báo lỗi
khi output bị cắt vì hết token hoặc model chưa tạo câu trả lời cuối.

`serve.py` giữ API tại `127.0.0.1:8000` và truyền key qua biến `VLLM_API_KEY`
của vLLM. Host kết nối bằng tunnel hoặc proxy ở các mục dưới. Script khởi
động vLLM trực tiếp; không có FastAPI trung gian tải thêm một bản model.
[Biến môi trường vLLM](https://docs.vllm.ai/en/stable/configuration/env_vars/)

### 2.1. Đổi model bằng `.env`

Ví dụ chuyển từ cấu hình Qwen3.8-27B-FP8 sang Qwen3.6-35B-A3B-FP8 để so sánh
trên GPU 48 GiB, sửa:

```dotenv
LLM_MODEL_ID=Qwen/Qwen3.6-35B-A3B-FP8
```

Giữ `LLM_SERVED_MODEL_NAME=rag-llm` ở server và `VLLM_MODEL=rag-llm` ở RAG.
Hai model này dùng chung cấu hình parser/thinking trong file mẫu. Kiểm tra
bộ nhớ và thời gian trả lời sau khi đổi; dung lượng tham số lớn hơn không tự
đồng nghĩa với chất lượng tốt hơn.

1. Trong terminal đang chạy server, `Ctrl+C` để dừng model cũ.
2. Sửa `.env`, chạy `python serve.py --dry-run` để xem đúng model và các tham số.
3. Chạy `python serve.py`; đợi tải/nạp model và khởi động API xong.
4. Ở terminal thứ hai, chạy `python check_api.py`, rồi thử `ask`/`chat` từ host.

Đổi file `.env` không thay model trong process đang chạy. Model mới vẫn phải
được vLLM hỗ trợ và vừa GPU/context đã chọn; `--dry-run` không xác nhận điều đó.
Không cần index lại tài liệu khi chỉ đổi LLM.

**Khi đổi sang họ model khác**, xem model card và kiểm tra các biến sau:

| Biến server | Cách dùng |
|---|---|
| `LLM_REASONING_PARSER` | Chọn parser đúng họ reasoning model; để trống nếu không dùng. Parser tách suy luận khỏi câu trả lời, không tự bật/tắt thinking. |
| `LLM_CHAT_TEMPLATE_KWARGS` | JSON object truyền cho chat template; ví dụ Qwen dùng `{"enable_thinking": false}`. Để trống để dùng mặc định model. |
| `LLM_LANGUAGE_MODEL_ONLY` | `true` nếu dùng multimodal model chỉ cho văn bản; `false` để bỏ flag này. |
| `LLM_QUANTIZATION` | Để trống để vLLM đọc cấu hình quantization từ model; chỉ điền khi hướng dẫn model yêu cầu, ví dụ `awq`. |
| `LLM_CHAT_TEMPLATE_FILE` | Để trống để dùng template của tokenizer; nếu cần ghi đè, điền file `.jinja` có sẵn. Đường tương đối tính từ module. |
| `LLM_DTYPE`, `LLM_MAX_MODEL_LEN`, `LLM_TENSOR_PARALLEL_SIZE` | Điều chỉnh theo GPU, số GPU và bộ nhớ cần cho model/context. |

Python không đoán parser từ tên model. Nếu không cấu hình, launcher không
thêm parser, template kwargs hay chế độ language-only. File `.env.example`
chọn rõ các giá trị dành cho Qwen, nên cần kiểm tra chúng khi đổi họ model.
[Reasoning và template defaults trong vLLM](https://docs.vllm.ai/en/stable/features/reasoning_outputs/)

Để model quyết định các tùy chọn riêng tại server, đặt `VLLM_THINK=` trống
trong `.env` RAG. `VLLM_THINK=true/false` ghi đè `enable_thinking` ở server;
chỉ dùng với chat template hỗ trợ tùy chọn đó. Giới hạn output của RAG vẫn
do `VLLM_MAX_TOKENS` điều khiển, độc lập với giới hạn checker.

**Nếu đã dùng bản cũ:** `.env` hiện có không tự thay đổi khi lấy code mới.
Bạn có thể giữ tên API `qwen3.5-4b`, miễn cả hai phía khớp nhau. Để dùng tên
ổn định mới, sửa cả `LLM_SERVED_MODEL_NAME` và `VLLM_MODEL` thành `rag-llm`
một lần. Với Qwen, thêm `LLM_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'`
ở server trước khi bỏ `VLLM_THINK=false` ở RAG, rồi khởi động lại server/RAG.

### 2.2. Dùng khoảng 43 GiB trên GPU 48 GiB dành riêng cho LLM

Đặt trong `.env` **server**:

```dotenv
LLM_GPU_MEMORY_UTILIZATION=0.90
```

Với GPU báo tổng 49140 MiB, tỷ lệ này tương ứng ngân sách khoảng **43.19 GiB**,
chừa khoảng **4.80 GiB** ngoài ngân sách. Đây là tỷ lệ trên **tổng VRAM**, không
phải phần đang trống. vLLM dùng ngân sách cho weights, bộ nhớ chạy model và
phần cache suy luận được tính sau khi đo bộ nhớ lúc khởi động. Số VRAM hiển thị
trên `nvidia-smi` có thể khác ngân sách vì cách cấp phát và các tiến trình khác.

`0.90` không giới hạn GPU ở 90% sức tính toán. Tăng từ `0.80` chủ yếu cho thêm
bộ nhớ cache; nó không tự tăng chất lượng model hoặc bảo đảm sinh từng token
nhanh hơn. Cache lưu kết quả tính toán của các token để phục vụ context dài
hơn, nhiều request đồng thời hoặc tái sử dụng phần đầu prompt khi có hỗ trợ.
[Tham số bộ nhớ vLLM](https://docs.vllm.ai/en/v0.29.0/cli/serve/#--gpu-memory-utilization),
[giới hạn của prefix caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/)

Để áp dụng cho instance đã chạy:

1. `Ctrl+C` trong terminal server cũ và chờ tiến trình nhả VRAM.
2. Sửa `.env`; giá trị `0.80` hoặc `0.70` đã ghi trong file vẫn ghi đè mặc định
   mới. Nếu đã export biến này trong shell, sửa hoặc unset nó vì shell ưu tiên.
3. Chạy `python serve.py --dry-run`, kiểm tra có `--gpu-memory-utilization 0.9`.
4. Chạy `python serve.py`. Ở terminal khác, chờ API sẵn sàng rồi chạy
   `python check_api.py` và `nvidia-smi`.

Giữ context 8192 và một request đồng thời cho lần đo đầu. Nếu cần lịch sử hoặc
tài liệu dài hơn, có thể thử `LLM_MAX_MODEL_LEN=16384`. Nếu thực sự có nhiều
request cùng lúc, thử `LLM_MAX_NUM_SEQS=4`. Thay từng biến và so thời gian trả
lời trên cùng bộ câu hỏi; tăng các giới hạn này không tự làm một request nhanh
hơn. Đo cả giai đoạn khởi động và khi nhận prompt dài để kiểm tra đủ bộ nhớ.

Nếu muốn chừa thêm bộ nhớ cho các đợt tải cao, `0.88` cho ngân sách khoảng
42.23 GiB trên GPU này. Không cần cố đạt đúng một con số trên `nvidia-smi` khi
tốc độ và khả năng phục vụ đã đáp ứng nhu cầu.

### 2.3. Model đáng thử trên một GPU 48 GiB

Danh sách tham khảo ngày **23/09/2026**, cho RAG văn bản tiếng Việt, context
8192 và một request đồng thời. Dung lượng dưới đây là **file trong repository**,
không phải tổng VRAM khi chạy; vẫn cần bộ nhớ cho cache và các buffer của vLLM.

| Model ID | Dung lượng tải xấp xỉ | Mục đích thử |
|---|---|---|
| [`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/tree/main) | 30.9 GB | Làm mốc so sánh chất lượng trong họ Qwen. Nếu đã chạy bản này, giữ lại kết quả trước khi đổi model. |
| [`google/gemma-4-31B-it-qat-w4a16-ct`](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct/tree/main) | 23.3 GB | Thử một họ model khác cho tiếng Việt; bản QAT 4-bit chính thức, định dạng compressed-tensors. |
| [`Qwen/Qwen3.6-35B-A3B-FP8`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8/tree/main) | 37.5 GB | So sánh tốc độ/chất lượng với model dense 27B; cần đo trên GPU thực tế. |

Qwen 35B-A3B là MoE: khoảng 35B tham số tổng, khoảng 3B được kích hoạt mỗi
token. Weights của toàn bộ model vẫn cần được nạp; số 35B không có nghĩa nó
tự động trả lời tốt hơn model dense 27B.
[Kiến trúc Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)

Gemma 4 31B có kết quả tiếng Việt tốt trên
[SEA-HELM ngày 18/09/2026](https://leaderboard.sea-lion.ai/detailed/VI).
Benchmark đó không xác nhận chất lượng của riêng bản QAT này trên dữ liệu bệnh
cây, và chưa có Qwen3.8 để đối chiếu. Vì vậy, dùng nó để chọn ứng viên thử,
không kết luận model nào tốt nhất cho RAG của dự án.

**Cấu hình chung để bắt đầu so sánh:**

```dotenv
LLM_GPU_MEMORY_UTILIZATION=0.90
LLM_MAX_MODEL_LEN=8192
LLM_MAX_NUM_SEQS=1
LLM_TENSOR_PARALLEL_SIZE=1
LLM_DTYPE=auto
LLM_LANGUAGE_MODEL_ONLY=true
LLM_QUANTIZATION=
LLM_CHAT_TEMPLATE_FILE=
LLM_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'
```

Với Qwen3.8-27B-FP8:

```dotenv
LLM_MODEL_ID=Qwen/Qwen3.8-27B-FP8
LLM_REASONING_PARSER=qwen3
```

Với Qwen3.6-35B-A3B-FP8, thay `LLM_MODEL_ID`; giữ parser `qwen3`.
Với Gemma 4 31B QAT, thay cả hai dòng:

```dotenv
LLM_MODEL_ID=google/gemma-4-31B-it-qat-w4a16-ct
LLM_REASONING_PARSER=gemma4
```

Giữ nguyên API key và tên `LLM_SERVED_MODEL_NAME` đang khớp với `VLLM_MODEL`
của RAG. Để trống quantization cho vLLM đọc định dạng từ checkpoint. Các ví dụ
dùng template đi kèm model, tắt thinking cho câu trả lời RAG thông thường.
Đặt `VLLM_THINK=` trống ở RAG để dùng mặc định server; giá trị `true` ở client
sẽ bật thinking lại cho request đó.
Tham khảo hướng dẫn runtime khi phiên bản vLLM hiện tại chưa hỗ trợ model:
[Qwen3.8](https://recipes.vllm.ai/Qwen/Qwen3.8-27B),
[Gemma 4](https://github.com/vllm-project/recipes/blob/main/Google/Gemma4.md).

Sau mỗi lần đổi, dừng server cũ rồi dry-run, khởi động và kiểm tra API như mục
2.1. Các cấu hình này chưa được benchmark trên instance Vast của bạn. Dùng cùng
20–30 câu hỏi, cùng các chunk truy xuất và giới hạn output để so sánh: đúng tài
liệu, không thêm thông tin ngoài nguồn, xử lý khi thiếu dữ liệu, thời gian chờ
token đầu và tốc độ sinh câu trả lời. Không tăng số chunk chỉ để dùng hết VRAM.

## 3. Kết nối RAG từ host bằng SSH tunnel

Đây là cách ít bước nhất để thử từ máy của bạn. Không cần tên miền, certificate
hay public cổng API của instance; chỉ cần SSH cho phép port forwarding.

### 3.1. Mở tunnel trên Windows

Lấy host, port, user và key từ lệnh SSH Vast cung cấp. Thay các placeholder
trong lệnh dưới; tên file key chỉ là ví dụ:

```powershell
ssh -i "$env:USERPROFILE\.ssh\id_ed25519_vast" -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -p SSH_PORT -L 127.0.0.1:8001:127.0.0.1:8000 root@SSH_HOST
```

Giữ terminal này mở. Windows `127.0.0.1:8001` được chuyển tới Vast
`127.0.0.1:8000`. Nếu đổi `LLM_PORT`, đổi cổng cuối trong `-L` tương ứng.
[SSH port forwarding trên Vast](https://docs.vast.ai/guides/instances/connect/ssh)

### 3.2. Chọn API remote trong RAG

Trên **host**, ghép các dòng trong [rag-client.env.example](rag-client.env.example)
vào `RAG-module/.env`. Giữ các thiết lập embedding, chunking và retrieval đang
dùng. Không copy `.env` của server đè lên `.env` của RAG.

```dotenv
LLM_PROVIDER=vllm
VLLM_BASE_URL=http://127.0.0.1:8001/v1
VLLM_HOST=
VLLM_PORT=
VLLM_MODEL=rag-llm
VLLM_API_KEY=DIEN_CUNG_KEY_VOI_SERVER
VLLM_MAX_TOKENS=800
VLLM_TIMEOUT=120
VLLM_THINK=
```

Sau đó, trong PowerShell mới:

```powershell
Set-Location D:\HCMUT\Capstone_Project\RAG-module
.\.venv\Scripts\Activate.ps1
python ..\LLM-server-module\check_api.py --env-file .env
python main.py search "Bệnh ghẻ trên cây táo xử lý như thế nào?" --debug
python main.py ask "Bệnh ghẻ trên cây táo xử lý như thế nào?"
python main.py chat --debug
```

`check_api.py` dùng URL/model/key trong `.env` RAG và không nạp embedding.
Nếu lệnh đó chạy được nhưng `ask` lỗi, kiểm tra cấu hình/index RAG riêng.
`search` vẫn chạy hoàn toàn ở host; `ask`/`chat` gửi prompt tới Vast.

**Không cần index lại chỉ vì chuyển LLM sang Vast.** Model embedding và
Chroma vẫn ở host. `chat` giữ embedding và lịch sử trong cùng process như cũ;
lịch sử hiện tại chưa lưu xuống disk.

Đổi về Ollama/Gemini chỉ cần sửa `LLM_PROVIDER` cùng config tương ứng rồi mở
lại RAG. Cấu hình `.env` không tự bật/tắt instance hoặc đổi model trên Vast.

## 4. Public API qua HTTPS (tùy chọn)

Nếu muốn host khác gọi bằng URL mà không giữ tunnel, dùng cấu hình
[nginx.conf.example](nginx.conf.example):

```text
Host RAG -> https://DOMAIN:PUBLIC_PORT/v1 -> Nginx :8443 -> vLLM 127.0.0.1:8000
```

Mục này cần tên miền và certificate hợp lệ cho tên miền đó. Proxy chỉ mở
`/v1/models` và `/v1/chat/completions`; backend vLLM kiểm tra API key. API key
của vLLM không bảo vệ mọi endpoint, nên giữ backend ở loopback như script.
[Phạm vi xác thực vLLM](https://docs.vllm.ai/en/stable/usage/security/)

1. Khi tạo instance/template trên Vast, thêm `-p 8443:8443` vào **Docker options
   trong giao diện Vast**. Không chạy `docker run` bên trong instance.
2. Sau khi instance lên, xem **IP Port Info**, lấy mapping
   `PUBLIC_IP:PUBLIC_PORT -> 8443/tcp`. Public port có thể khác 8443.
3. Trỏ tên miền về IP public đó và chuẩn bị certificate. Cách cấp/gia hạn
   certificate phụ thuộc nhà cung cấp DNS; các lệnh dưới dùng certificate đã có.

[Port mapping của Vast.ai](https://docs.vast.ai/guides/instances/connect/networking)

Trên Linux, cài Nginx bằng root/sudo, rồi tạo cấu hình runtime:

```bash
apt-get install -y nginx
cd /workspace/capstone-project-039-253/LLM-server-module
mkdir -p runtime /workspace/llm-tls
if [ ! -f runtime/nginx.conf ]; then
  cp nginx.conf.example runtime/nginx.conf
fi
nano runtime/nginx.conf
```

Sửa `server_name`, đặt certificate chain và private key đúng đường dẫn trong
config. Nếu đổi `LLM_PORT`, sửa cả hai `proxy_pass`. Kiểm tra và chạy proxy
trong cửa sổ tmux riêng:

```bash
nginx -t -c /workspace/capstone-project-039-253/LLM-server-module/runtime/nginx.conf
nginx -c /workspace/capstone-project-039-253/LLM-server-module/runtime/nginx.conf -g 'daemon off;'
```

Thay URL trong `.env` RAG; giữ nguyên model và API key:

```dotenv
VLLM_BASE_URL=https://DOMAIN:PUBLIC_PORT/v1
```

Chạy lại `check_api.py --env-file .env` từ thư mục RAG trên host. Khi public,
kiểm tra request không có key tới `/v1/models` bị từ chối và `/invocations`
trả 404. Chỉ backend RAG giữ key; không đưa key vào frontend trình duyệt.

Nếu RAG của bạn chạy trong container, URL public vẫn dùng được. Với tunnel,
`127.0.0.1` trong container là chính container; phải cấu hình địa chỉ tới host
đang mở tunnel thay cho địa chỉ loopback trên Windows.

## 5. Chạy lại, cập nhật và xử lý lỗi

- **Mất SSH:** SSH lại, dùng `tmux ls` rồi attach phiên đang chạy. Nếu dùng
  tunnel, mở lại tunnel ở host. LLM còn chạy thì không nạp model lại.
- **Stop/start instance:** activate `.venv`, chạy lại `python serve.py` và proxy
  nếu dùng. Cache còn trên disk thì không tải weights lại. Tmux không giữ
  tiến trình qua reboot.
- **Đổi model/context:** sửa `.env` server, `Ctrl+C` server cũ, rồi chạy lại.
  Nếu đổi tên API, sửa `VLLM_MODEL` bên RAG cho khớp.
- **Lấy code mới:** dừng server trước, `git status --short`, rồi `git pull --ff-only`;
  dùng `bash install.sh` khi requirements đổi, hoặc `bash install.sh --upgrade`
  khi muốn nâng cấp các package đang cài, rồi khởi động lại server.
- **Kết thúc thuê:** thoát terminal không dừng tính phí. Stop còn phí disk;
  destroy xóa disk instance. Sao lưu `.env`, cache/config runtime cần giữ
  trước khi destroy. [Vòng đời instance](https://docs.vast.ai/guides/instances/manage-instances)

| Lỗi | Cách kiểm tra |
|---|---|
| `Connection refused` | Server đã sẵn sàng chưa, tunnel/proxy có chạy và đúng port không? |
| HTTP 401/403 | `VLLM_API_KEY` ở RAG phải khớp `LLM_API_KEY` ở server. |
| HTTP 404 / sai model | URL có `/v1`; `VLLM_MODEL` phải khớp `LLM_SERVED_MODEL_NAME`. |
| HTTP 301/302/307/308 | Sửa URL sang endpoint HTTPS cuối cùng; checker không tự chuyển tiếp API key qua redirect. |
| CUDA không hoạt động | Kiểm tra `nvidia-smi`, đúng venv và driver/wheel theo tài liệu vLLM. |
| OOM | Kiểm tra GPU còn model khác, giảm context/concurrency hoặc dùng model nhỏ hơn; giảm mức VRAM không tự chữa mọi lỗi thiếu bộ nhớ. |
| Không hỗ trợ `qwen3_5` hoặc flag | Kiểm tra phiên bản vLLM/model card; cập nhật trong venv server riêng. |
| Đổi `.env` chưa có tác dụng | Mở lại process; biến shell đã export có ưu tiên hơn `.env`. |
| Retrieval vẫn sai | Kiểm tra `search --debug` ở host; GPU LLM remote không tự sửa retrieval. |

vLLM giữ LLM trong GPU suốt thời gian server chạy. `OLLAMA_KEEP_ALIVE` không áp
dụng cho vLLM. Context limit phải đủ cho prompt + lịch sử + các chunk + output.

## 6. Kiểm tra code khi sửa

Trong venv có `python-dotenv`, từ `LLM-server-module`:

```bash
python -m unittest discover -s tests -v
```

Các test kiểm tra cấu hình, cách khởi động, không lộ key trong argv và HTTP
request/response. Có test HTTP thật trên loopback để kiểm tra auth và chặn
redirect; API trong test mô phỏng phản hồi LLM. Test installer dùng công cụ
giả trong thư mục tạm, không tải packages. Không cần cài vLLM/GPU để chạy test này. Test RAG
riêng kiểm tra request thật của LangChain qua HTTP transport giả tới endpoint
HTTPS remote.

Đã đối chiếu source và tài liệu chính thức ngày 23/09/2026. Chưa kiểm thử tải
model/inference hoặc Nginx trên instance Vast.ai thực tế; `check_api.py` là bước
kiểm tra môi trường thật sau khi bạn triển khai.
