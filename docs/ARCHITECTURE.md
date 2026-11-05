# Kiến trúc hệ thống

Tài liệu này mô tả chi tiết kỹ thuật của bốn giai đoạn trong pipeline. Con số và tham số phải khớp với `configs/*.yaml` và với slide thuyết trình.

---

## GĐ1 — Tiền xử lý & xây chỉ mục

**Chạy một lần, ~2–3 giờ.** Không phụ thuộc truy vấn.

### 1.1 Doc Store

Nguồn: `metadata.csv` của CORD-19 bản 19/05/2020. Mỗi dòng là một bài báo.

| Trường | Dùng để |
|---|---|
| `cord_uid` | khoá chính, là đơn vị truy xuất |
| `title` | trường truy xuất (trọng số ×2) |
| `abstract` | trường truy xuất |
| `authors`, `journal`, `publish_time`, `doi`, `pmcid` | trích dẫn nguồn trên UI |

**Quan trọng:** Doc Store giữ `abstract` **nguyên một chuỗi**. Không lưu `sentences[]` — việc tách câu làm lúc truy vấn, chỉ cho 10 bài hiển thị (xem ADR 0003).

Bài **chỉ có title, không có abstract vẫn được giữ** — có thể nằm trong qrels, nên loại đi là mất recall.

Văn bản đưa vào truy xuất: `title + abstract`. **Không dùng full-text** (bài toán là truy xuất mức bài báo, full-text làm loãng tín hiệu và tăng thời gian encode).

### 1.2 Chỉ mục BM25

- Lucene qua `pyserini` (dự phòng `bm25s` khi máy không có JDK — cùng ngữ nghĩa, xem ADR [0006](adr/0006-mot-truong-van-ban-lap-title.md)).
- **Một trường văn bản**: `title` lặp 2 lần rồi tới `abstract` — **xấp xỉ** `title² + abstract`, không phải fielded BM25 thật (lặp title làm `|d|` dài ra nên ảnh hưởng hệ số `b`).
- `EnglishAnalyzer`: stemming Porter + bỏ stopword tiếng Anh.
- `k1 = 0.9`, `b = 0.4` — hiệu chỉnh trên tập dev (mặc định Lucene là 1.2 / 0.75; CORD-19 có abstract dài và nhiều thuật ngữ lặp nên `b` thấp hơn cho kết quả tốt hơn).
- Lấy **1000 ứng viên** mỗi truy vấn — đủ sâu cho mọi độ đo.

### 1.3 Chỉ mục vector

- Mô hình: **MedCPT** (NCBI) — dùng **cặp** `Query-Encoder` + `Article-Encoder` (bi-encoder). Hai checkpoint này phải đi cùng nhau; trộn với checkpoint khác sẽ ra vector ở hai không gian khác nhau.
- Article Encoder encode **toàn bộ** corpus (~170k bài), 768 chiều, cắt ở 512 token.
- Chuẩn hoá **L2 vector bài báo** rồi mới nạp vào `IndexFlatIP` → tích vô hướng bằng cosine.
- `IndexFlatIP` **không có bước train** — khác với `IndexIVFFlat`. Ghi ra đĩa, nạp lại khi chạy.
- Kích thước: 170k × 768 × 4 byte ≈ **500 MB**.
- Encode một lần ~2–3 giờ (GPU), cache xuống đĩa.

> Không nhầm lẫn: **FAISS là chỉ mục**, không phải mô hình. Encoder tạo vector, FAISS chỉ tìm vector gần nhất.

---

## GĐ2 — Truy xuất đa nhánh

**Mỗi truy vấn, < 1 giây.** Hai nhánh chạy song song (hoặc tuần tự, tổng vẫn dưới ngưỡng).

| Nhánh | Cơ chế | Mạnh ở | Yếu ở |
|---|---|---|---|
| BM25 | trùng khớp từ chính xác | tên thuốc, mã gen, mã thử nghiệm, biến thể | viết tắt, đồng nghĩa |
| Dense | tương đồng vector | diễn đạt lại, câu hỏi tự nhiên | token hiếm bị chẻ sub-word rồi nhoè |

Mỗi nhánh trả `(docid, score)` cho **top-1000**.

Hai nhánh phải **độc lập** — không để nhánh này lọc ứng viên cho nhánh kia, nếu không mất tính đa dạng mà fusion cần.

---

## GĐ3 — Dung hợp + rerank

### 3.1 Dung hợp ở mức bài báo

Ba công thức, chạy trên **cùng một tập ứng viên** (hợp của hai danh sách top-1000):

```
(1) RRF chuẩn          RRF(d) = Σ 1 / (k + rank(d)),          k = 60
(2) Weighted RRF       RRF(d) = Σ w / (k + rank(d)),           w = trọng số mỗi nhánh
(3) Convex sau min-max s(d)   = α·bm25(d) + (1−α)·dense(d),    điểm đã chuẩn hoá [0, 1]
```

Vai trò của từng công thức — đây là **thiết kế thí nghiệm**, không phải ba lựa chọn ngang nhau:

| Công thức | Trả lời câu hỏi |
|---|---|
| (1) RRF chuẩn | Đường cơ sở. Chỉ dùng thứ hạng, không có gì tinh chỉnh ngoài `k`. |
| (2) Weighted RRF | Một nhánh có đáng tin hơn nhánh kia không? (`Σw = 1`, học trên dev) |
| (3) Convex | Biên độ điểm có mang thông tin mà thứ hạng làm mất không? (`α` học trên dev) |

`min-max` phải tính **trên tập ứng viên của từng truy vấn**, không phải trên toàn corpus.

### 3.2 Rerank

- Mô hình: **MedCPT Cross-Encoder** — checkpoint **thứ ba**, khác hai checkpoint trên.
- Input: cặp `(query, title + abstract)`, cắt 512 token.
- Chấm **top-50** sau dung hợp (không phải top-1000 — cross-encoder chạy 170k cặp sẽ mất hàng giờ).
- Output là **một điểm** cho mỗi cặp, không phải vector → **không cache được**.

> **Bắt buộc:** sau rerank phải ghi điểm **đơn điệu giảm** theo thứ hạng cuối. Nếu để lẫn thang điểm (cross-encoder 0–1 và BM25 0–30) trong cùng một cột `score` của run file, `trec_eval` sẽ sắp xếp lại theo `score` và **đảo ngược thứ hạng** bạn vừa tính.

---

## GĐ4 — Hậu xử lý

### 4.1 Lọc residual

Run của vòng 3 chỉ được chứa tài liệu nằm trong `docids-rnd3.txt` (tập tài liệu hợp lệ của vòng — đã loại các bài đã chấm ở vòng 1–2). Tài liệu ngoài tập này bị loại **trước khi** ghi run file, không phải sau.

Lý do: TREC-COVID dùng **residual collection** — mỗi vòng chỉ chấm trên phần tài liệu chưa được chấm. Trả về tài liệu đã chấm ở vòng trước là vi phạm thể lệ và làm điểm không so sánh được.

### 4.2 Trích dẫn mức câu (snippet)

Chạy **sau khi** đã có thứ hạng cuối, và chỉ cho **trang đang hiển thị** (10 bài).

```
ranking cuối
   └─► 10 bài của trang hiện tại
         ├─ lấy abstract từ Doc Store
         ├─ tách câu           (~10 ms cho 10 abstract)
         ├─ lọc câu rác        (quá ngắn, chỉ có số, bản quyền, link trơ)
         ├─ chấm ~50 cặp (query, câu)  → chọn câu điểm cao nhất mỗi bài
         └─ tô sáng từ khoá khớp query
```

Nguyên tắc bất biến:

1. **Nguyên văn** — không sửa một chữ nào trong câu.
2. **Không cắt giữa câu** — quá dài thì cắt ở ~40 từ + `…`, nhưng không cắt mất từ khoá.
3. **Không trộn bài** — câu của bài A chỉ xuất hiện dưới tiêu đề bài A.
4. **Không sinh văn bản** — không dùng LLM tóm tắt/diễn giải.

Tách câu phải xử lý viết tắt y sinh: `et al.`, `i.e.`, `e.g.`, `Fig.`, `No.`, `vs.`, `Dr.` — chỉ cắt khi ký tự đầu sau khoảng trắng là **chữ hoa** và từ trước không nằm trong danh sách viết tắt.

---

## Đánh giá

- **Công cụ:** `trec_eval` cục bộ qua `pytrec_eval-terrier` (không nộp Kaggle, ADR 0009). nDCG dùng gain tuyến tính (gain = nhãn).
- **Chính:** `nDCG@10` — độ đo chuẩn của TREC-COVID.
- **Phụ:** `nDCG@20`, `MAP`, `P@5`, `bpref`.
- **Run file:** 6 cột `topicid Q0 docid rank score run-tag`.

Lưu ý:

- Phán định là **không đầy đủ** (pooling). Tài liệu chưa được chấm bị coi là **không liên quan**.
- `qrels` chỉ có cho **topic 1–35**; 5 topic mới (36–40) chưa có phán định → điểm trung bình tính trên 35 topic.
- `trec_eval` sắp xếp lại theo cột `score` và **bỏ qua** cột `rank`.
- Mặc định `trec_eval` **không** tính nDCG — phải chỉ định `-m ndcg_cut.10`.
- Không dùng cờ `-c` khi chấm 40 topic: nó kéo các topic không có phán định vào trung bình với điểm 0.

---

## Sơ đồ phụ thuộc module & Quy ước dữ liệu

```
config ──► io ──► preprocess ──► index ──► retrieve ──► fuse ──► rerank ──► postfilter
                                                                             │
                                                        evaluate ◄───────────┤
                                                        web ◄────────────────┤
                                                        snippet ◄────────────┘
```

- **Quy tắc phụ thuộc:** Luồng dữ liệu một chiều. Module sau được import module trước; không được import ngược.
- **Quy ước dữ liệu chung (Data Contracts):** Để các thành viên phát triển độc lập không bị xung đột khi tích hợp, tất cả các module sử dụng chung các dataclass định nghĩa tại [`src/trec_covid/schemas.py`](../src/trec_covid/schemas.py). Chi tiết đặc tả đầu vào / đầu ra và dữ liệu mẫu (mock data) được quy định tại [`docs/DATA_CONTRACTS.md`](DATA_CONTRACTS.md).

