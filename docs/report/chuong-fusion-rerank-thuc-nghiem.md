# Chương: Dung hợp, tái xếp hạng và thực nghiệm

> **Bản nháp (2026-11-05). Người viết: Người 3.** Mọi con số lấy từ `results/2026-11-05_*.csv`, đo
> với run dense **tạm**: Người 3 tự encode bằng MedCPT Article/Query Encoder để không phải chờ
> nhánh của Người 2. Khi có `runs/dense.txt` chính thức, chạy lại
> `07` (nếu 07 báo tham số lệch thì sửa `configs/fusion.yaml`) `→ 04 → 04 --postfilter … → 08 → 09 → 10`
> rồi cập nhật các bảng có đánh dấu †.

## 1. Dung hợp hai nhánh ở mức bài báo

Mỗi truy vấn có hai danh sách top-1000: BM25 và Dense (MedCPT). Thang điểm của hai danh sách
không so sánh được với nhau. Điểm BM25 không chuẩn hoá và phụ thuộc `k1`, `b`, độ dài văn
bản. Điểm cosine của MedCPT co cụm trong khoảng hẹp. Ba công thức dưới đây chạy trên **cùng một
tập ứng viên** (hợp hai danh sách, tối đa ~2.000 bài), rồi cắt lại **top-1000** theo hợp đồng
dữ liệu.

| Công thức | Định nghĩa | Câu hỏi thí nghiệm |
|---|---|---|
| RRF chuẩn | `s(d) = Σ 1 / (k + rank(d))`, `k = 60` | Đường cơ sở: chỉ dùng thứ hạng |
| Weighted RRF | `s(d) = Σ wᵢ / (k + rankᵢ(d))`, `w_bm25 + w_dense = 1` | Có nhánh nào đáng tin hơn không? |
| Convex | `s(d) = α·bm25'(d) + (1 − α)·dense'(d)` | Biên độ điểm có mang thông tin mà thứ hạng làm mất không? |

**Chuẩn hoá min-max theo từng truy vấn.** Với Convex, `bm25'` và `dense'` được min-max trên
**chính danh sách của truy vấn đó**, không dùng thang lý thuyết `[-1, 1]` của cosine. Điểm MedCPT
thực tế chỉ trải trong một khoảng hẹp. Nếu chuẩn hoá theo `[-1, 1]` thì mọi chênh lệch giữa
các bài bị nén lại, và Convex sẽ gần như chỉ còn BM25. Bài vắng mặt ở một nhánh nhận 0 ở nhánh
đó. Bài cuối của mỗi danh sách cũng nhận 0 sau min-max. Đây là tính chất vốn có của min-max,
và chỉ ảnh hưởng bài ở hạng 1000.

**Thứ tự và phá hoà.** trec_eval sắp lại run theo điểm giảm dần và phá hoà theo docid giảm
dần, bỏ qua cột `rank`. Mọi tầng của hệ thống xếp hạng theo đúng khoá này, để thứ hạng hệ thống
tính ra trùng với thứ hạng trec_eval chấm.

## 2. Tái xếp hạng bằng Cross-Encoder

Top-50 sau dung hợp được chấm lại bằng **MedCPT Cross-Encoder** (`ncbi/MedCPT-Cross-Encoder`).
Đây là checkpoint thứ ba, khác cặp Query/Article Encoder của nhánh Dense. Đầu vào là cặp
`[truy vấn, title + abstract]`, cắt 512 token. Bài không có abstract thì chỉ dùng title.
Cross-encoder đọc truy vấn và bài **cùng lúc** nên chính xác hơn bi-encoder, nhưng không
tính trước được vector. Vì vậy chỉ chấm 50 bài chứ không chấm cả 1000.

**Ghép điểm đơn điệu.** Cross-encoder trả **logit** (có thể âm), không phải xác suất. Nếu ghi
logit cho top-50 và điểm fusion cho phần còn lại vào cùng một cột `score`, trec_eval sẽ sắp lại
theo điểm và trộn hai phần với nhau. Hệ thống gán cho phần đầu điểm
`tail_max + 1 + (ce − ce_min)`, trong đó `tail_max` là điểm fusion cao nhất của phần đuôi.
Như vậy phần đầu luôn cao hơn hẳn phần đuôi, còn thứ tự bên trong phần đầu giữ đúng thứ tự
của cross-encoder. Sau mỗi tầng, hàm `check_ranking` kiểm tra ba bất biến: điểm không tăng,
không trùng docid, rank liền mạch từ 1. Sai bất biến nào thì **báo lỗi**, không tự sắp lại để
che lỗi.

## 3. Hậu xử lý: lọc residual

TREC-COVID chấm theo *residual collection*: mỗi vòng chỉ chấm những bài chưa được chấm ở vòng
trước. Đếm trực tiếp trên dữ liệu (ADR 0008) cho thấy:

- `docids-rnd3.txt` có 128.162 id duy nhất, **trùng khít** Doc Store. Đây là **danh sách được
  phép** của vòng 3, không phải danh sách bài đã chấm.
- 9.822 trên 10.287 bài đã chấm ở vòng 1–2 vẫn nằm trong danh sách đó. 465 bài có phán định đã
  bị gỡ khỏi CORD-19 bản 19/05/2020. Không hệ thống nào của nhóm truy hồi được chúng, nên đây là
  trần recall cố định.

Hệ thống cài ba chế độ postfilter (ADR 0005): `off`, `docids` (giữ bài trong danh sách được
phép) và `judged` (residual đúng nghĩa: bỏ bài đã có phán định cho chính topic đó).

## 4. Thiết lập thực nghiệm

| Hạng mục | Thiết lập |
|---|---|
| Collection | CORD-19 bản 19/05/2020, 128.162 bài (title + abstract) |
| Truy vấn | 40 topic vòng 3, trường `query`; chấm trên **35 topic** có qrels |
| Phán định | `qrels.csv`: 20.728 phán định của **vòng 1–2** (iteration 0.5–2.0), nhãn 0/1/2 |
| BM25 † | `bm25s`, `k1 = 0.9`, `b = 0.4`, title lặp 2 lần (ADR 0006) |
| Dense † | MedCPT Article/Query Encoder, CLS, chuẩn hoá L2, FAISS `IndexFlatIP` (run tạm) |
| Chấm điểm | `pytrec_eval-terrier`, bọc đúng mã C của trec_eval (ADR 0009) |
| Độ đo | nDCG@10 (chính), nDCG@20, MAP, P@5, bpref; trung bình chỉ trên topic có qrels (tương đương không `-c`) |
| Phần cứng | NVIDIA RTX 2000 Ada Laptop 8 GB, rerank fp16 |

**Lưu ý về nDCG của trec_eval.** trec_eval dùng gain **tuyến tính** (gain = nhãn 0/1/2), không
phải `2^rel − 1` như một số tài liệu khác.

**Phán định không đầy đủ.** Bài chưa có phán định bị coi là không liên quan. Mà qrels chỉ có
vòng 1–2, được gom (pool) từ các run nộp vòng 1–2. Hệ thống tìm ra bài mới mà pool cũ chưa từng
thấy sẽ bị phạt oan. Đo **judged@10** (tỉ lệ bài trong top-10 có phán định) cho thấy độ lệch
này rất lớn: BM25 đạt 0,43, Dense chỉ 0,21. Vì vậy nhóm báo thêm hai độ đo:

- **bpref**: ít nhạy với bài chưa chấm.
- **nDCG@10 condensed** (Sakai 2007): bỏ bài chưa chấm khỏi run, xếp lại rồi mới chấm. Độ đo
  này chỉ so cách hệ thống sắp các bài *đã* có nhãn, nên công bằng giữa các nhánh khi qrels
  thưa.

**Tuning không học trên tập chấm (ADR 0010).** Chỉ có 35 topic và không có tập dev riêng. Tham
số `w_bm25` và `α` được quét trên lưới {0,0; 0,1; …; 1,0} bằng **5-fold cross-validation theo
topic** (seed 42): mỗi fold chọn tham số tốt nhất trên 4 fold còn lại rồi chấm trên fold của
mình. Độ đo để chọn là nDCG@10 condensed. Số báo cáo cho công thức có tham số là điểm CV.

## 5. Kết quả

### 5.1. So sánh ba công thức dung hợp †

`results/2026-11-05_fusion-comparison.csv`

| Run | nDCG@10 | nDCG@10 condensed | CV 5-fold | MAP | bpref | judged@10 | Tham số |
|---|---|---|---|---|---|---|---|
| BM25 | 0,2155 | 0,4601 | — | 0,0609 | 0,2648 | 0,43 | |
| Dense | 0,1387 | 0,5943 | — | 0,0437 | 0,2960 | 0,21 | |
| RRF | 0,2068 | 0,5730 | 0,5730 | 0,0732 | 0,3248 | 0,35 | k = 60 |
| Weighted RRF | 0,1869 | 0,6019 | 0,5899 | 0,0659 | 0,3288 | 0,30 | w_bm25 = 0,4 |
| Convex | 0,1801 | 0,6014 | **0,5937** | 0,0633 | **0,3380** | 0,28 | α = 0,3 |

- Hai độ đo **cho thứ tự ngược nhau**. Theo nDCG@10 thường, BM25 dẫn đầu và Dense kém nhất.
  Theo nDCG@10 condensed thì ngược lại (Dense 0,594 so với BM25 0,460). Nếu tuning theo nDCG@10
  thường, Weighted RRF chọn `w_bm25 = 1,0`, tức là bỏ hẳn Dense. Đó là hệ quả của pool vòng
  1–2, không phản ánh chất lượng.
- Đường nDCG@10 condensed theo tham số đạt đỉnh ở `w_bm25 = 0,4` (0,6019) và `α = 0,3`
  (0,6014), rồi giảm đều về phía BM25 thuần (0,4601 ở 1,0). Cả 5 fold đều chọn tham số trong
  khoảng 0,2–0,4, nên kết quả tuning ổn định.
- Convex có điểm CV cao nhất (0,5937), nên được chọn làm `fusion.active`. Tuy vậy, trên
  nDCG@10 condensed, dung hợp mới chỉ **ngang** Dense đơn lẻ (0,5937 so với 0,5943; thắng 13
  topic, thua 18). Dung hợp hơn rõ ở bpref (0,338 so với 0,296) và MAP (0,063 so với 0,044),
  tức là ở độ sâu lớn hơn top-10.

### 5.2. Ablation 5 tầng †

`results/2026-11-05_ablation.csv`. BM25 và Dense là hai nhánh song song nên không có Δ. Δ của
Fusion là so với nhánh đơn tốt nhất theo từng độ đo: BM25 với nDCG@10 thường, Dense với nDCG@10
condensed. Các dòng sau so với dòng ngay trước. ‡ Fusion và các dòng sau dùng α = 0,3 chọn trên cả
35 topic (in-sample), nên lạc quan hơn số CV ở mục 5.1 (0,5937).

| Tầng | nDCG@10 | Δ | nDCG@10 condensed | Δ | MAP | P@5 | bpref |
|---|---|---|---|---|---|---|---|
| BM25 | 0,2155 | | 0,4601 | | 0,0609 | 0,2629 | 0,2648 |
| Dense | 0,1387 | | 0,5943 | | 0,0437 | 0,1829 | 0,2960 |
| Fusion (Convex) ‡ | 0,1801 | −0,0354 | 0,6014 | +0,0071 | 0,0633 | 0,2286 | 0,3380 |
| + Rerank ‡ | **0,2230** | +0,0429 | **0,6289** | +0,0276 | **0,0707** | 0,2343 | **0,3382** |
| + Residual (docids) | 0,2230 | 0 | 0,6289 | 0 | 0,0707 | 0,2343 | 0,3382 |
| + Residual (judged) | 0 | −0,2230 | 0 | −0,6289 | 0 | 0 | 0 |

- Dung hợp một mình **thấp hơn BM25** trên nDCG@10 thường (−0,0354). Trên condensed, nó chỉ hơn
  Dense +0,0071 khi dùng tham số in-sample, còn với số CV thì thấp hơn Dense (0,5937 so với 0,5943).
- **Rerank là tầng đóng góp rõ nhất** theo cả hai độ đo. Pipeline đầy đủ đạt nDCG@10 cao nhất
  (0,2230) và nDCG@10 condensed cao nhất (0,6289). Trên condensed, nó thắng Dense đơn lẻ ở 19
  topic và thua ở 13.
- Lọc `docids` không bỏ bài nào, nên năm cột đầu của run file giống hệt run trước nó.
- Lọc `judged` bỏ toàn bộ 3.950 bài có nhãn nằm trong top-1000, gồm 2.045 bài liên quan, nên mọi
  độ đo bằng 0. Đây là **thước đo mất nhãn**, không phải hệ thống kém. Vì vậy mặc định là
  `off` (ADR 0005), và chế độ `judged` không được bật trên giao diện.

### 5.3. Độ trễ

`results/2026-11-05_latency.csv`, 40 truy vấn, bỏ 3 lần khởi động

| Tầng | median (ms) | p95 (ms) |
|---|---|---|
| BM25 (`bm25s`) | 3,4 | 5,4 |
| Fusion (Convex) | 2,9 | 4,5 |
| Rerank top-50 (fp16, GPU) | 231,3 | 275,2 |
| Toàn pipeline (chưa gồm Dense) † | 273,3 | 326,7 |
| Dense † | — | — |

Rerank chiếm gần hết thời gian xử lý nhưng vẫn dưới ngưỡng 2 giây đặt ra. Thời gian encode truy
vấn và tìm FAISS của nhánh Dense sẽ được bổ sung khi có `search_dense()` của Người 2.

### 5.4. Tái lập

`scripts/10_check_reproducible.py` lưu manifest gồm các độ đo làm tròn 4 chữ số và mã băm thứ
tự top-10 của từng topic. Điểm số không được băm, vì fp16 trên GPU khác nhau có thể lệch ở chữ
số cuối mà không đổi thứ hạng. Chạy lại rerank trên cùng máy cho kết quả **khớp 100%**, thứ tự
docid giống hệt. Phép thử trên máy thứ hai (task 1.9.3) sẽ cập nhật sau †.

## 6. Thảo luận và giới hạn

1. **Đánh giá với qrels vòng 1–2 thiên vị BM25.** judged@10 của Dense chỉ bằng một nửa của
   BM25. Mọi kết luận "nhánh A tốt hơn nhánh B" phải đọc kèm judged@10, bpref và nDCG@10
   condensed. Đây cũng là lý do điểm nDCG@10 tuyệt đối (~0,22) thấp hơn nhiều so với các con
   số thường thấy cho TREC-COVID khi dùng đủ qrels của 5 vòng.
2. **Run Dense là bản tạm.** Kết luận "dung hợp chỉ ngang Dense trên condensed" có thể đổi khi
   có run chính thức.
3. **465 bài có phán định không có trong collection** là trần recall cố định của mọi cấu hình.
4. **`k = 60` của RRF không được tune**, nhóm giữ theo Cormack và cộng sự (2009) để RRF đúng
   vai trò đường cơ sở.
5. **Rerank 50 bài** là thỏa hiệp giữa chất lượng và thời gian chạy. Chưa đo với 30 hoặc 100
   bài.

## 7. Câu hỏi phản biện liên quan

- **"nDCG@10 = 0,22 nghĩa là gì, sao thấp vậy?"** Đây là nDCG@10 trên qrels vòng 1–2. Trung bình
  hơn một nửa top-10 chưa từng được chấm và bị tính là không liên quan. Khi chỉ chấm trên các bài
  đã có nhãn (condensed), pipeline đạt 0,63.
- **"Vì sao chỉ 35 topic?"** qrels chỉ có phán định cho topic 1–35. Topic 36–40 bị bỏ khỏi trung
  bình (tương đương trec_eval không dùng `-c`), không tính là 0.
- **"Cross-encoder có làm chậm người dùng không?"** Median 231 ms cho 50 bài trên GPU laptop. Toàn
  pipeline chưa gồm Dense mất 273 ms.
- **"Có tái lập được không?"** Có manifest và script so sánh. Trên cùng máy, kết quả khớp 100%.
- **"Vì sao không lọc residual?"** Bản chất của residual là bỏ đúng các bài có nhãn vòng 1–2,
  nên chấm cục bộ sẽ ra 0 (đã đo). Danh sách `docids-rnd3.txt` thì là danh sách được phép và
  không loại bài nào.

## Tài liệu tham khảo

- Cormack, G. V., Clarke, C. L. A., & Büttcher, S. (2009). Reciprocal rank fusion outperforms
  Condorcet and individual rank learning methods. *SIGIR 2009*.
- Sakai, T. (2007). Alternatives to bpref. *SIGIR 2007*.
- Jin, Q., et al. (2023). MedCPT: Contrastive pre-trained transformers with large-scale PubMed
  search logs for zero-shot biomedical information retrieval. *Bioinformatics*.
- Voorhees, E., et al. (2020). TREC-COVID: Constructing a pandemic information retrieval test
  collection. *SIGIR Forum*.
