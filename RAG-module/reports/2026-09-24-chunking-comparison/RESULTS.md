# Kết quả so sánh chunking (tự sinh)

Sinh bởi `scripts/compare_chunking.py` lúc 2026-09-24T11:18:15+00:00. Đừng sửa tay: chạy lại script sẽ ghi đè.
Phần nhận xét nằm ở `README.md`.

## 1. Thiết lập

| Cấu hình | Dữ liệu | Chunking | Số tài liệu | Fingerprint dữ liệu |
| --- | --- | --- | ---: | --- |
| A. Dữ liệu cũ + recursive | artifacts/chunking/baseline/data | recursive 1.000 ký tự, overlap 150 | 22 | 20c4cd3eb68b |
| B. Dữ liệu mới + recursive | data | recursive 1.000 ký tự, overlap 150 | 25 | 7139bf71038b |
| C. Dữ liệu mới + structure | data | structure ≤ 400 token, overlap 40 | 25 | 7139bf71038b |

- Bộ eval: `eval/chunking_queries.jsonl`, 60 câu, tính điểm trên **58 câu có đáp án** (40 dev, 18 heldout; 59 đoạn bằng chứng). 2 câu ngoài corpus bị loại khỏi điểm.
- Embedding `AITeamVN/Vietnamese_Embedding` (cuda); token đếm bằng tokenizer của embedding.
- Retrieval chính: hybrid (semantic + BM25, RRF k=60, 20 ứng viên mỗi nhánh), không reranker; hệ thống dùng k = 4.
- Khoảng tin cậy 95%: bootstrap 10.000 lần theo câu hỏi (seed 20260924).
- Git `7689b4f+thay đổi chưa commit`; Python 3.11.0, langchain-core 1.6.3, chromadb 1.5.9.

## 2. Đặc trưng index

| Độ đo | A. Dữ liệu cũ + recursive | B. Dữ liệu mới + recursive | C. Dữ liệu mới + structure |
| --- | ---: | ---: | ---: |
| Số chunk | 362 | 420 | 743 |
| Chunk / tài liệu | 16,5 | 16,8 | 29,7 |
| Token / chunk: trung bình | 346 | 335 | 168 |
| Token / chunk: trung vị | 360 | 346 | 150 |
| Token / chunk: P5–P95 | 230–397 | 185–403 | 94–318 |
| Token / chunk: lớn nhất | 429 | 431 | 399 |
| Độ lệch chuẩn token | 51 | 60 | 67 |
| Tỉ lệ token dành cho header | 25,2% | 24,3% | 30,8% |
| Chunk > 512 token | 0,0% | 0,0% | 0,0% |
| Hệ số lặp (token thân chunk / token corpus) | 1,12 | 1,11 | 0,90 |
| Phủ corpus (ký tự nằm trong thân ≥ 1 chunk) | 100,0% | 100,0% | 91,8% |
| Phủ nội dung (không tính dòng heading, trường metadata) | 100,0% | 100,0% | 100,0% |
| Bắt đầu ở đầu dòng/câu | 100,0% | 100,0% | 97,4% |
| Kết thúc ở cuối dòng/câu | 100,0% | 100,0% | 100,0% |
| Không cắt ngang câu (cả hai đầu) | 100,0% | 100,0% | 97,4% |
| Nằm trong đúng 1 mục H2 | — | 41,7% | 100,0% |
| Nằm trong đúng 1 mục lá | — | 17,1% | 100,0% |
| Số mục H2 trung bình / chunk | — | 1,81 | 1,00 |
| Bằng chứng nằm trọn trong 1 chunk | 96,6% | 94,9% | 100,0% |
| Số chunk chạm 1 bằng chứng (TB) | 1,22 | 1,27 | 1,00 |
| Thời gian build index (s) | 12,7 | 14,1 | 15,1 |
| Dung lượng index (MB) | 7,6 | 8,8 | 11,1 |

Mục H2/mục lá chỉ đo được trên dữ liệu mới (dữ liệu cũ không có heading marker).
Thời gian build gồm tách chunk, embedding và ghi Chroma; model đã nạp sẵn trước khi bấm giờ.

## 3. Retrieval chính (hybrid, k = 4)

| Độ đo | A. Dữ liệu cũ + recursive | B. Dữ liệu mới + recursive | C. Dữ liệu mới + structure |
| --- | ---: | ---: | ---: |
| Hit@1 | 0,586 [0,466–0,707] | 0,569 [0,448–0,690] | 0,672 [0,552–0,793] |
| Recall@4 | 0,845 [0,741–0,931] | 0,931 [0,862–0,983] | 0,871 [0,784–0,948] |
| Coverage@4 | 0,863 [0,777–0,940] | 0,953 [0,897–0,995] | 0,871 [0,776–0,948] |
| MRR@10 | 0,717 [0,621–0,806] | 0,720 [0,631–0,806] | 0,763 [0,672–0,851] |
| nDCG@4 | 0,757 [0,665–0,840] | 0,811 [0,741–0,873] | 0,780 [0,688–0,863] |
| nDCG@10 | 0,802 [0,734–0,867] | 0,836 [0,783–0,888] | 0,806 [0,727–0,881] |
| Context precision@4 | 0,267 [0,233–0,302] | 0,289 [0,263–0,319] | 0,220 [0,198–0,237] |
| Mật độ bằng chứng@4 | 0,049 [0,041–0,057] | 0,054 [0,047–0,061] | 0,093 [0,078–0,109] |
| Doc Hit@1 | 0,983 [0,948–1,000] | 1,000 [1,000–1,000] | 0,966 [0,914–1,000] |
| Doc Recall@4 | 1,000 [1,000–1,000] | 1,000 [1,000–1,000] | 1,000 [1,000–1,000] |
| Doc precision@4 | 0,875 [0,823–0,922] | 0,888 [0,845–0,927] | 0,828 [0,776–0,879] |
| Recall@1024 token | 0,767 [0,655–0,871] | 0,819 [0,716–0,905] | 0,888 [0,802–0,966] |
| Recall@1536 token | 0,828 [0,724–0,914] | 0,931 [0,862–0,983] | 0,940 [0,871–0,991] |

| Chi phí | A. Dữ liệu cũ + recursive | B. Dữ liệu mới + recursive | C. Dữ liệu mới + structure |
| --- | ---: | ---: | ---: |
| Token ngữ cảnh top-4: trung bình | 1 414 | 1 374 | 805 |
| Token ngữ cảnh top-4: trung vị | 1 417 | 1 382 | 779 |
| Token ngữ cảnh top-4: P95 | 1 532 | 1 529 | 1 116 |
| Chunk đúng tài liệu nhưng sai mục (TB / 4) | 2,43 | 2,40 | 2,43 |
| Độ trễ retrieval: trung vị (ms) | 23,1 | 27,7 | 34,0 |
| Độ trễ retrieval: P95 (ms) | 28,1 | 31,1 | 40,8 |
| Required context recall@4 (4 câu) | 100,0% | 100,0% | 100,0% |

Số trong ngoặc vuông là khoảng tin cậy 95%.

## 4. Theo số chunk k (hybrid)

| k | Recall A | Recall B | Recall C | Coverage A | Coverage B | Coverage C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0,586 | 0,569 | 0,672 | 0,606 | 0,630 | 0,672 |
| 2 | 0,759 | 0,767 | 0,741 | 0,783 | 0,798 | 0,741 |
| 3 | 0,802 | 0,888 | 0,845 | 0,820 | 0,903 | 0,845 |
| 4 | 0,845 | 0,931 | 0,871 | 0,863 | 0,953 | 0,871 |
| 5 | 0,897 | 0,948 | 0,922 | 0,925 | 0,971 | 0,922 |
| 6 | 0,948 | 0,948 | 0,940 | 0,959 | 0,971 | 0,940 |
| 8 | 0,983 | 1,000 | 0,940 | 0,983 | 1,000 | 0,940 |
| 10 | 0,983 | 1,000 | 0,940 | 0,983 | 1,000 | 0,940 |

## 5. Cùng ngân sách token ngữ cảnh (hybrid)

Lấy chunk theo thứ hạng cho tới khi chunk kế tiếp làm vượt ngân sách. So sánh này công bằng hơn so cùng k, vì chunk của các cấu hình dài ngắn khác nhau.

| Ngân sách | Recall A | Recall B | Recall C | Số chunk TB A | Số chunk TB B | Số chunk TB C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 0,000 | 0,000 | 0,517 | 0,0 | 0,0 | 0,9 |
| 512 | 0,586 | 0,569 | 0,810 | 1,0 | 1,0 | 2,1 |
| 768 | 0,741 | 0,716 | 0,871 | 1,9 | 1,9 | 3,4 |
| 1 024 | 0,767 | 0,819 | 0,888 | 2,2 | 2,4 | 4,7 |
| 1 536 | 0,828 | 0,931 | 0,940 | 4,0 | 4,1 | 7,5 |
| 2 048 | 0,914 | 0,948 | 0,940 | 5,3 | 5,5 | 10,2 |
| 3 072 | 0,983 | 1,000 | 0,974 | 8,3 | 8,6 | 15,6 |

## 6. Kiểm định theo cặp (hybrid)

Nhị phân: McNemar chính xác; liên tục: Wilcoxon signed-rank. p Holm hiệu chỉnh cho 3 phép so sánh của cùng một độ đo. Chênh lệch = cấu hình sau − cấu hình trước, kèm khoảng tin cậy 95% bootstrap.

| Độ đo | So sánh | Trước | Sau | Chênh lệch [KTC 95%] | Tốt hơn/Kém hơn/Bằng | p | p Holm |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hit@1 | A→B (Viết lại dữ liệu (cùng recursive)) | 0,586 | 0,569 | -0,017 [-0,138; 0,103] | 6/7/45 | 1,000 | 1,000 |
| Hit@1 | B→C (Đổi cách chunk (cùng dữ liệu mới)) | 0,569 | 0,672 | 0,103 [-0,034; 0,224] | 11/5/42 | 0,210 | 0,630 |
| Hit@1 | A→C (Tổng hợp cả hai) | 0,586 | 0,672 | 0,086 [-0,034; 0,207] | 9/4/45 | 0,267 | 0,630 |
| Doc Hit@1 | A→B (Viết lại dữ liệu (cùng recursive)) | 0,983 | 1,000 | 0,017 [0,000; 0,052] | 1/0/57 | 1,000 | 1,000 |
| Doc Hit@1 | B→C (Đổi cách chunk (cùng dữ liệu mới)) | 1,000 | 0,966 | -0,034 [-0,086; 0,000] | 0/2/56 | 0,500 | 1,000 |
| Doc Hit@1 | A→C (Tổng hợp cả hai) | 0,983 | 0,966 | -0,017 [-0,069; 0,034] | 1/2/55 | 1,000 | 1,000 |
| Recall@4 | A→B (Viết lại dữ liệu (cùng recursive)) | 0,845 | 0,931 | 0,086 [0,000; 0,172] | 6/1/51 | 0,059 | 0,176 |
| Recall@4 | B→C (Đổi cách chunk (cùng dữ liệu mới)) | 0,931 | 0,871 | -0,060 [-0,164; 0,043] | 3/7/48 | 0,292 | 0,584 |
| Recall@4 | A→C (Tổng hợp cả hai) | 0,845 | 0,871 | 0,026 [-0,078; 0,138] | 6/5/47 | 0,560 | 0,584 |
| MRR@10 | A→B (Viết lại dữ liệu (cùng recursive)) | 0,717 | 0,720 | 0,003 [-0,081; 0,085] | 15/11/32 | 0,730 | 0,949 |
| MRR@10 | B→C (Đổi cách chunk (cùng dữ liệu mới)) | 0,720 | 0,763 | 0,043 [-0,058; 0,146] | 12/13/33 | 0,474 | 0,949 |
| MRR@10 | A→C (Tổng hợp cả hai) | 0,717 | 0,763 | 0,046 [-0,040; 0,134] | 17/9/32 | 0,242 | 0,725 |
| nDCG@10 | A→B (Viết lại dữ liệu (cùng recursive)) | 0,802 | 0,836 | 0,033 [-0,018; 0,085] | 18/12/28 | 0,178 | 0,533 |
| nDCG@10 | B→C (Đổi cách chunk (cùng dữ liệu mới)) | 0,836 | 0,806 | -0,030 [-0,099; 0,037] | 13/14/31 | 0,524 | 1,000 |
| nDCG@10 | A→C (Tổng hợp cả hai) | 0,802 | 0,806 | 0,003 [-0,064; 0,067] | 18/11/29 | 0,626 | 1,000 |
| Coverage@4 | A→B (Viết lại dữ liệu (cùng recursive)) | 0,863 | 0,953 | 0,089 [0,023; 0,171] | 6/1/51 | 0,023 | 0,068 |
| Coverage@4 | B→C (Đổi cách chunk (cùng dữ liệu mới)) | 0,953 | 0,871 | -0,082 [-0,176; 0,005] | 3/7/48 | 0,064 | 0,128 |
| Coverage@4 | A→C (Tổng hợp cả hai) | 0,863 | 0,871 | 0,007 [-0,097; 0,112] | 6/6/46 | 0,869 | 0,869 |
| Context precision@4 | A→B (Viết lại dữ liệu (cùng recursive)) | 0,267 | 0,289 | 0,022 [-0,017; 0,060] | 11/7/40 | 0,275 | 0,275 |
| Context precision@4 | B→C (Đổi cách chunk (cùng dữ liệu mới)) | 0,289 | 0,220 | -0,069 [-0,099; -0,039] | 1/17/40 | 0,000 | 0,000 |
| Context precision@4 | A→C (Tổng hợp cả hai) | 0,267 | 0,220 | -0,047 [-0,086; -0,013] | 5/16/37 | 0,016 | 0,033 |
| Mật độ bằng chứng@4 | A→B (Viết lại dữ liệu (cùng recursive)) | 0,049 | 0,054 | 0,005 [0,001; 0,009] | 29/27/2 | 0,365 | 0,365 |
| Mật độ bằng chứng@4 | B→C (Đổi cách chunk (cùng dữ liệu mới)) | 0,054 | 0,093 | 0,039 [0,027; 0,052] | 49/8/1 | 0,000 | 0,000 |
| Mật độ bằng chứng@4 | A→C (Tổng hợp cả hai) | 0,049 | 0,093 | 0,043 [0,031; 0,057] | 50/6/2 | 0,000 | 0,000 |
| Recall@1024 token | A→B (Viết lại dữ liệu (cùng recursive)) | 0,767 | 0,819 | 0,052 [-0,052; 0,155] | 6/3/49 | 0,317 | 0,412 |
| Recall@1024 token | B→C (Đổi cách chunk (cùng dữ liệu mới)) | 0,819 | 0,888 | 0,069 [-0,034; 0,172] | 7/3/48 | 0,206 | 0,412 |
| Recall@1024 token | A→C (Tổng hợp cả hai) | 0,767 | 0,888 | 0,121 [0,000; 0,241] | 10/3/45 | 0,052 | 0,157 |
| Recall@1536 token | A→B (Viết lại dữ liệu (cùng recursive)) | 0,828 | 0,931 | 0,103 [0,017; 0,190] | 7/1/50 | 0,034 | 0,069 |
| Recall@1536 token | B→C (Đổi cách chunk (cùng dữ liệu mới)) | 0,931 | 0,940 | 0,009 [-0,069; 0,086] | 3/3/52 | 0,739 | 0,739 |
| Recall@1536 token | A→C (Tổng hợp cả hai) | 0,828 | 0,940 | 0,112 [0,017; 0,216] | 8/2/48 | 0,023 | 0,069 |
| Token ngữ cảnh top-4 | A→B (Viết lại dữ liệu (cùng recursive)) | 1 414 | 1 374 | -40 [-62; -18] | 17/41/0 | 0,001 | 0,001 |
| Token ngữ cảnh top-4 | B→C (Đổi cách chunk (cùng dữ liệu mới)) | 1 374 | 805 | -569 [-630; -510] | 0/58/0 | 0,000 | 0,000 |
| Token ngữ cảnh top-4 | A→C (Tổng hợp cả hai) | 1 414 | 805 | -609 [-662; -555] | 0/58/0 | 0,000 | 0,000 |

Với Token ngữ cảnh, "tốt hơn" nghĩa là nhiều token hơn (tức tốn hơn).

## 7. Theo tập dev/heldout (hybrid)

| Tập | A. Dữ liệu cũ + recursive: Hit@1 / Recall@4 / MRR | B. Dữ liệu mới + recursive: Hit@1 / Recall@4 / MRR | C. Dữ liệu mới + structure: Hit@1 / Recall@4 / MRR |
| --- | ---: | ---: | ---: |
| dev (n=40) | 0,550 / 0,850 / 0,697 | 0,550 / 0,975 / 0,728 | 0,675 / 0,875 / 0,771 |
| heldout (n=18) | 0,667 / 0,833 / 0,761 | 0,611 / 0,833 / 0,702 | 0,667 / 0,861 / 0,745 |

## 8. Theo loại câu hỏi (hybrid, Hit@1 / Recall@4)

| Loại | A. Dữ liệu cũ + recursive | B. Dữ liệu mới + recursive | C. Dữ liệu mới + structure |
| --- | ---: | ---: | ---: |
| acronym (n=3) | 1,00 / 1,00 | 1,00 / 1,00 | 0,67 / 0,67 |
| confusable (n=4) | 0,25 / 0,75 | 0,50 / 1,00 | 0,50 / 1,00 |
| exact_topic (n=13) | 0,38 / 0,92 | 0,46 / 1,00 | 0,54 / 0,77 |
| formula (n=1) | 1,00 / 1,00 | 1,00 / 1,00 | 1,00 / 1,00 |
| mixed_language (n=5) | 0,80 / 1,00 | 1,00 / 1,00 | 1,00 / 1,00 |
| multi_fact (n=2) | 1,00 / 1,00 | 1,00 / 1,00 | 1,00 / 1,00 |
| multi_host (n=1) | 0,00 / 1,00 | 0,00 / 1,00 | 0,00 / 1,00 |
| multi_source (n=1) | 0,00 / 1,00 | 0,00 / 1,00 | 0,00 / 0,50 |
| negation (n=4) | 0,75 / 0,75 | 0,50 / 0,75 | 0,75 / 1,00 |
| no_accent (n=2) | 0,50 / 0,50 | 0,00 / 1,00 | 0,50 / 0,50 |
| paraphrase (n=10) | 0,30 / 0,50 | 0,40 / 0,70 | 0,40 / 0,80 |
| qualifier (n=10) | 1,00 / 1,00 | 0,70 / 1,00 | 1,00 / 1,00 |
| symptom_only (n=2) | 0,50 / 1,00 | 0,50 / 1,00 | 1,00 / 1,00 |

Mỗi loại chỉ có vài câu: dùng để tìm điểm yếu, không để kết luận.

## 9. Ảnh hưởng của phương pháp tìm (k = 4)

| Mode | Độ đo | A. Dữ liệu cũ + recursive | B. Dữ liệu mới + recursive | C. Dữ liệu mới + structure |
| --- | --- | ---: | ---: | ---: |
| hybrid | Hit@1 | 0,586 | 0,569 | 0,672 |
| hybrid | Recall@4 | 0,845 | 0,931 | 0,871 |
| hybrid | MRR@10 | 0,717 | 0,720 | 0,763 |
| hybrid | nDCG@10 | 0,802 | 0,836 | 0,806 |
| semantic | Hit@1 | 0,483 | 0,569 | 0,586 |
| semantic | Recall@4 | 0,784 | 0,853 | 0,776 |
| semantic | MRR@10 | 0,635 | 0,697 | 0,679 |
| semantic | nDCG@10 | 0,737 | 0,797 | 0,736 |
| bm25 | Hit@1 | 0,534 | 0,552 | 0,569 |
| bm25 | Recall@4 | 0,776 | 0,819 | 0,828 |
| bm25 | MRR@10 | 0,656 | 0,669 | 0,681 |
| bm25 | nDCG@10 | 0,747 | 0,778 | 0,743 |

## 10. Định nghĩa độ đo

- **Bằng chứng**: đoạn trích nguyên văn trong tài liệu, gán nhãn cho từng câu hỏi. Một chunk *chứa trọn*
  bằng chứng khi mọi ký tự khác khoảng trắng của đoạn đó nằm trong khoảng [start, end) của chunk.
- **Hit@1**: chunk xếp hạng 1 chứa trọn ít nhất một bằng chứng.
- **Recall@k**: tỉ lệ bằng chứng được phủ trọn bởi hợp của k chunk đầu (một bằng chứng có thể nằm vắt qua hai chunk).
- **Coverage@k**: như Recall@k nhưng tính điểm từng phần theo tỉ lệ ký tự được phủ.
- **MRR@10**: nghịch đảo thứ hạng của chunk đầu tiên chứa trọn một bằng chứng (0 nếu không có trong top 10).
- **nDCG@k**: độ lợi của một chunk = tỉ lệ bằng chứng mà riêng chunk đó phủ; chuẩn hóa theo thứ tự lý tưởng
  của mọi chunk trong index.
- **Context precision@4**: tỉ lệ chunk trong top-4 chạm vào bằng chứng hoặc ngữ cảnh bắt buộc. Thiên vị chunk
  to: chunk càng dài càng dễ chạm bằng chứng, và bằng chứng vắt qua hai chunk được đếm hai lần.
- **Mật độ bằng chứng@4**: tỉ lệ ký tự trong thân top-4 chunk là bằng chứng/ngữ cảnh đã gán nhãn (precision
  mức ký tự); phần chữ thừa bị tính vào mẫu số nên không thiên vị chunk to.
- **Doc Hit@1 / Doc Recall@4 / Doc precision@4**: như trên nhưng ở mức tài liệu (đúng file nguồn).
- **Recall@N token**: lấy chunk theo thứ hạng cho tới khi tổng token vượt N; đo recall trên phần đã lấy.
- **Chunk đúng tài liệu nhưng sai mục**: chunk thuộc tài liệu đúng nhưng không chạm bằng chứng/ngữ cảnh.
- **Required context recall@4**: tỉ lệ đoạn ngữ cảnh bắt buộc (điều kiện, ngoại lệ) được phủ trọn trong top-4.
- **Hệ số lặp**: tổng token thân chunk / tổng token corpus; > 1 do overlap, < 1 khi dòng heading và trường
  metadata nằm trong header thay vì thân chunk.
- **Phủ corpus / phủ nội dung**: tỉ lệ ký tự khác khoảng trắng nằm trong thân ít nhất một chunk; bản "nội dung"
  không tính dòng heading và thân các trường metadata (structure đưa chúng vào header/metadata của chunk).
- **Không cắt ngang câu**: chunk bắt đầu ở đầu dòng hoặc sau dấu kết câu, và kết thúc ở cuối dòng hoặc sau dấu kết câu.
- **Nằm trong 1 mục H2 / mục lá**: phần thân chunk chỉ chạm một mục `##` / một mục nhỏ nhất theo heading.

