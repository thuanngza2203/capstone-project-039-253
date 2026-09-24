# So sánh chunking: dữ liệu cũ/mới, recursive/structure (24/09/2026)

Thư mục này chứa kết quả đo cho báo cáo. Số liệu đầy đủ nằm ở [RESULTS.md](RESULTS.md) (tự sinh);
file này là phần diễn giải. Mọi con số ở đây lấy từ RESULTS.md.

## 1. Câu hỏi và thiết kế

| Cấu hình | Dữ liệu | Chunking | Số tài liệu | Số chunk |
| --- | --- | --- | ---: | ---: |
| **A** | trước tái cấu trúc | recursive (1.000 ký tự, overlap 150) | 22 | 362 |
| **B** | sau tái cấu trúc | recursive (1.000 ký tự, overlap 150) | 25 | 420 |
| **C** | sau tái cấu trúc | structure (≤ 400 token theo heading, overlap 40) | 25 | 743 |

- **A → B**: tác động của việc viết lại dữ liệu, giữ nguyên cách chunk.
- **B → C**: tác động của cách chunk, giữ nguyên dữ liệu.
- **A → C**: tác động tổng hợp (hệ thống cũ so với hệ thống hiện tại).

Ba index được build lại bằng cùng một phiên bản code, cùng embedding `AITeamVN/Vietnamese_Embedding`,
cùng retrieval hybrid (semantic + BM25, RRF), không reranker. Đánh giá trên 58 câu có nhãn bằng chứng
trong `eval/chunking_queries.jsonl` (40 dev, 18 heldout, 13 loại câu hỏi). Bằng chứng là đoạn trích
nguyên văn, khớp được ở cả dữ liệu cũ lẫn mới, nên cùng một nhãn dùng được cho cả ba cấu hình.

## 2. Tóm tắt

1. **Viết lại dữ liệu chủ yếu tăng độ phủ, không tăng độ chính xác top-1.** Recall@4 tăng từ 0,845
   lên 0,931 (6 câu tốt hơn, 1 câu kém hơn; p = 0,059) và Coverage@4 từ 0,863 lên 0,953 (p = 0,023).
   Hit@1 gần như không đổi (0,586 → 0,569).
2. **Structure xếp đúng đoạn lên đầu tốt hơn và gửi ít ngữ cảnh hơn nhiều.** So với B: Hit@1 tăng
   từ 0,569 lên 0,672 (11 câu tốt hơn, 5 câu kém hơn), MRR@10 từ 0,720 lên 0,763. Token ngữ cảnh top-4
   giảm 41% (1.374 → 807). Mật độ bằng chứng trong ngữ cảnh tăng khoảng 70% (0,054 → 0,092; p < 0,001).
3. **Cùng 4 chunk, structure phủ bằng chứng kém hơn recursive**: Recall@4 0,871 so với 0,931, vì mỗi
   chunk structure chỉ bằng khoảng một nửa chunk recursive.
4. **Cùng ngân sách ngữ cảnh, structure bằng hoặc hơn.** Ở 512 token: 0,810 so với 0,569; ở 1.024
   token: 0,888 so với 0,819; ở 1.536 token: 0,940 so với 0,931. Từ khoảng 2.000 token trở lên, hai
   cách gần như ngang nhau.
5. **Về thống kê**, với 58 câu thì chỉ các khác biệt về *ngữ cảnh* có ý nghĩa sau hiệu chỉnh Holm: số
   token, mật độ bằng chứng, và Context precision@4 (theo chiều ngược lại, do thiên vị chunk to, xem 4.2).
   Các khác biệt về chất lượng xếp hạng có hướng nhất quán nhưng chưa đủ ý nghĩa (Hit@1 B→C: p = 0,21).

## 3. Kết quả chính (hybrid, k = 4)

| Độ đo | A | B | C |
| --- | ---: | ---: | ---: |
| Hit@1 | 0,586 | 0,569 | **0,672** |
| MRR@10 | 0,717 | 0,720 | **0,763** |
| Recall@4 | 0,845 | **0,931** | 0,871 |
| Coverage@4 | 0,863 | **0,953** | 0,871 |
| nDCG@10 | 0,802 | **0,836** | 0,806 |
| Recall ở 1.024 token | 0,767 | 0,819 | **0,888** |
| Recall ở 1.536 token | 0,828 | 0,931 | **0,940** |
| Mật độ bằng chứng@4 | 0,049 | 0,054 | **0,092** |
| Token ngữ cảnh top-4 (trung vị) | 1.417 | 1.382 | **790** |
| Doc Hit@1 (đúng tài liệu ở hạng 1) | 0,983 | **1,000** | 0,966 |

Khoảng tin cậy 95% và các độ đo khác: RESULTS.md mục 3. Biểu đồ: `figures/main_metrics.png`,
`figures/recall_at_budget.png`, `figures/cost_density.png`.

## 4. Diễn giải

### 4.1 Viết lại dữ liệu (A → B)

Dữ liệu mới có heading rõ ràng và đoạn văn gọn hơn. Cùng cách chunk recursive, bằng chứng được phủ
trọn trong top-4 thường xuyên hơn (Recall@4 +0,086; Coverage@4 +0,089, KTC 95% [0,023; 0,171]).
Thứ hạng của đoạn đúng nhất không đổi (MRR 0,717 → 0,720). Nghĩa là dữ liệu tốt hơn giúp đoạn liên quan
lọt vào top-4, nhưng không giúp recursive đưa đúng đoạn lên hạng 1.

Lưu ý: recursive không dùng heading khi cắt. Trên dữ liệu mới chỉ 41,7% chunk recursive nằm gọn
trong một mục `##`, trung bình mỗi chunk trải qua 1,81 mục (RESULTS.md mục 2).

### 4.2 Đổi cách chunk (B → C)

Structure cắt theo heading nên 100% chunk nằm trong đúng một mục, và 100% đoạn bằng chứng nằm trọn trong
một chunk (recursive: 94,9%). Hệ quả:

- **Hạng 1 chính xác hơn**: đoạn đúng không bị pha nội dung của mục khác, nên dễ xếp đầu hơn.
- **Ngữ cảnh gọn hơn**: 4 chunk structure chỉ khoảng 800 token so với khoảng 1.380 của recursive.
  Tỉ lệ ngữ cảnh là bằng chứng gần gấp đôi, tức LLM đọc ít chữ thừa hơn.
- **Phủ ít hơn ở cùng k**: mỗi chunk nhỏ hơn nên 4 chunk phủ ít văn bản hơn. Đây là lý do Recall@4
  của C thấp hơn B, và chỉ có C là có câu không tìm thấy bằng chứng ngay cả ở top-10
  (Recall@10: 0,940 so với 1,000).

So cùng số chunk thì thiệt cho structure. So cùng ngân sách token, tức cùng chi phí gửi cho LLM, thì
structure bằng hoặc hơn recursive ở mọi mức tới khoảng 1.536 token (`figures/recall_at_budget.png`).

Context precision@4 (tỉ lệ chunk chạm bằng chứng) của C thấp hơn B (0,220 so với 0,289). Chỉ số này thiên
vị chunk to: chunk dài dễ chạm bằng chứng hơn, và bằng chứng vắt qua hai chunk recursive được đếm hai lần.
Mật độ bằng chứng (precision mức ký tự) không có thiên vị đó và cho kết quả ngược lại. Trong báo cáo nên
dùng mật độ bằng chứng; nếu nêu Context precision@4 thì cần giải thích thiên vị này.

### 4.3 Theo phương pháp tìm (RESULTS.md mục 9)

| Recall@4 | A | B | C |
| --- | ---: | ---: | ---: |
| chỉ semantic | 0,784 | 0,853 | 0,776 |
| chỉ BM25 | 0,776 | 0,819 | 0,828 |
| hybrid | 0,845 | 0,931 | 0,871 |

Với chunk nhỏ, BM25 bắt từ khóa tốt hơn (C cao nhất ở BM25), còn embedding cần nhiều ngữ cảnh hơn
(C thấp nhất ở semantic). Hybrid tốt nhất ở cả ba cấu hình.

### 4.4 Các trường hợp structure trượt

| Câu | Loại | Hiện tượng |
| --- | --- | --- |
| q013 "chay la ngo phuong bac vet benh dai bao nhieu" | no_accent | Chunk đúng ở hạng 16; hạng 1 là tài liệu cà chua. B tìm được trong top-4 |
| q048 đốm vi khuẩn trên lá đào mới xuất hiện ướt hay khô | exact_topic | Chunk đúng ở hạng 11 |
| q058 so sánh thối đen và ghẻ trên lá táo | multi_source | Hạng 1 là `grape_black_rot.txt`, một tài liệu **mới thêm** cùng tên "thối đen"; chỉ tìm được 1/2 bằng chứng |
| q052 vì sao gọi là cháy lá (dâu tây) | paraphrase | Cả B và C đều trượt |

Câu gõ không dấu là điểm yếu rõ nhất của chunk nhỏ: ít chữ để embedding bù cho phần BM25 không khớp dấu.

## 5. Hạn chế

- **Cỡ mẫu nhỏ**: 58 câu, mỗi loại câu chỉ 1–13 câu. Khác biệt Hit@1 cỡ 10 điểm phần trăm cần nhiều câu
  hơn để có ý nghĩa thống kê.
- **Nhãn do nhóm tự soạn** từ corpus, chưa được chuyên gia nông nghiệp duyệt.
- **Tập dev đã được dùng khi thiết kế structure.** Trên heldout (18 câu), Hit@1 của C (0,667) bằng A
  (0,667) và chỉ hơn B (0,611) một câu. Phần lớn lợi thế Hit@1 của C nằm ở tập dev (0,675 so với 0,550),
  nên có thể một phần là do đã tinh chỉnh theo dev.
- **Dữ liệu mới có thêm 3 tài liệu** (`grape_black_rot`, `grape_leaf_blight`, `potato_late_blight`) mà bộ
  eval không hỏi tới. Chúng làm B và C khó hơn A (tài liệu gây nhiễu, ví dụ q058), nên so sánh A→B và
  A→C là thận trọng.
- **Chỉ đo retrieval, chưa đo câu trả lời.** Chất lượng câu trả lời của LLM cần đánh giá riêng, bằng khảo
  sát người dùng (`experiments/`) hoặc LLM chấm điểm.
- **Hit@1 nghiêm ngặt**: phải chứa trọn đoạn bằng chứng. Một chunk chứa 90% bằng chứng vẫn tính là trượt
  (Coverage@k cho điểm từng phần).
- **Độ trễ** đo trên một máy có GPU, mỗi câu chạy một lần; chênh lệch vài ms không đáng kể so với thời
  gian LLM sinh câu trả lời.

## 6. Khuyến nghị

- Giữ **dữ liệu mới + structure**, nhưng tăng `RAG_TOP_K` lên **6**. Recall@6 của C đạt 0,940, bằng mức
  tối đa C đạt được trong top-10, với khoảng 1.200 token ngữ cảnh (ước từ 807 token cho 4 chunk), vẫn ít
  hơn recursive top-4 (khoảng 1.380).
- Cải thiện câu không dấu: chuẩn hóa bỏ dấu cho BM25, hoặc bật reranker để kéo chunk đúng lên từ top-20.
- Mở rộng bộ eval (thêm câu heldout, câu cho 3 tài liệu mới) trước khi khẳng định kết luận về Hit@1.

## 7. File trong thư mục

| File | Nội dung |
| --- | --- |
| `RESULTS.md` | Toàn bộ bảng: đặc trưng index, retrieval, theo k, theo ngân sách, kiểm định, theo loại câu, theo mode, định nghĩa độ đo |
| `summary.json` | Toàn bộ số liệu tổng hợp và thông tin tái lập (git, phiên bản, fingerprint dữ liệu, cấu hình) |
| `index_stats.csv` | Đặc trưng từng index |
| `retrieval_summary.csv` | Độ đo tổng hợp theo cấu hình × mode, kèm khoảng tin cậy |
| `significance.csv` | Kiểm định theo cặp (McNemar, Wilcoxon, bootstrap, Holm) |
| `per_query.csv` | Điểm từng câu × cấu hình × mode (để vẽ lại hoặc phân tích thêm) |
| `evidence_containment.csv` | Từng đoạn bằng chứng có nằm trọn trong một chunk không |
| `chunks.csv` | Từng chunk: token, vị trí, cắt ngang câu/mục |
| `top_hits.jsonl` | Top-4 của từng câu (hybrid), để trích ví dụ định tính |
| `figures/` | `main_metrics`, `recall_at_k`, `recall_at_budget`, `cost_density`, `chunk_tokens`, `recall_by_category` (PNG, 200 dpi) |

CSV lưu UTF-8 có BOM để Excel đọc đúng tiếng Việt.

## 8. Tái lập

```powershell
cd RAG-module
$env:HF_HUB_OFFLINE = "1"          # nếu embedding đã có trong cache
python scripts/compare_chunking.py                   # build lại 3 index vào artifacts/ rồi đo
python scripts/compare_chunking.py --reuse-indexes   # chỉ đo lại
```

Script không đụng `chroma_db/` hay `chroma_db_structure/`. Biểu đồ cần `matplotlib`
(`python -m pip install matplotlib`); thiếu thì script vẫn ghi đủ bảng và CSV.
