# Phân công nhóm

Điền tên thật vào bảng dưới khi bắt đầu làm.

| Mã | Họ tên | MSSV | Vai trò | Giai đoạn |
|---|---|---|---|---|
| **A** | ……………… | ………… | Dữ liệu & chỉ mục | GĐ1 |
| **B** | ……………… | ………… | Truy xuất hai nhánh | GĐ2 |
| **C** | ……………… | ………… | Dung hợp & rerank | GĐ3 |
| **D** | ……………… | ………… | Trích dẫn, đánh giá, ứng dụng | GĐ4 |

Nhóm trưởng: ……………… (luân phiên theo tuần nếu cần)

Giảng viên hướng dẫn: ………………

---

## 1. Phạm vi từng người

Cách chia bám theo **bốn giai đoạn của pipeline**, nên mỗi người sở hữu một tầng trọn vẹn: biết input là gì, output là gì, và có thể tự chạy end-to-end trên tầng của mình.

### A — Dữ liệu & chỉ mục (GĐ1)

| Sở hữu | File / thư mục |
|---|---|
| Đọc CORD-19, dựng Doc Store | `src/trec_covid/io/` |
| Chuẩn hoá văn bản | `src/trec_covid/preprocess/` |
| Chỉ mục Lucene BM25 | `src/trec_covid/index/bm25_index.py` |
| Chỉ mục FAISS + encode MedCPT | `src/trec_covid/index/dense_index.py` |
| Script 01–03 | `scripts/01_…`, `02_…`, `03_…` |
| `configs/base.yaml`, `configs/bm25.yaml`, `configs/dense.yaml` | |

**Bàn giao:** Doc Store đã chuẩn hoá + 2 chỉ mục nạp được trong < 10 giây.

**Ước lượng:** nặng nhất về thời gian chờ (encode 2–3 giờ), nhẹ về logic.

### B — Truy xuất hai nhánh (GĐ2)

| Sở hữu | File / thư mục |
|---|---|
| Truy vấn BM25 | `src/trec_covid/retrieve/bm25.py` |
| Truy vấn dense | `src/trec_covid/retrieve/dense.py` |
| Chuẩn hoá truy vấn (dùng chung) | `src/trec_covid/preprocess/text.py` |
| `configs/` phần retrieve | |

**Bàn giao:** hàm `search(query) -> list[(docid, score)]` cho mỗi nhánh, chạy < 500 ms cho top-1000.

**Phụ thuộc:** cần chỉ mục của A. Trong lúc chờ, dựng trước bằng chỉ mục giả trên 1000 bài.

### C — Dung hợp & rerank (GĐ3)

| Sở hữu | File / thư mục |
|---|---|
| Ba công thức fusion | `src/trec_covid/fuse/rrf.py` |
| Cross-Encoder | `src/trec_covid/rerank/cross_encoder.py` |
| `configs/fusion.yaml`, `configs/rerank.yaml` | |

**Bàn giao:** hàm `fuse(lists) -> ranked` và `rerank(query, candidates) -> ranked`, kèm script thí nghiệm so sánh 3 công thức.

**Ước lượng:** nặng nhất về logic và thí nghiệm — đây là phần tạo ra điểm số.

### D — Trích dẫn, đánh giá, ứng dụng (GĐ4)

| Sở hữu | File / thư mục |
|---|---|
| Tách câu + chấm câu + tô sáng | `src/trec_covid/snippet/` |
| Lọc residual, ghi run file | `src/trec_covid/postfilter/` |
| Wrapper trec_eval, ablation | `src/trec_covid/evaluate/` |
| Ứng dụng Flask | `src/trec_covid/web/` |
| `configs/snippet.yaml`, `requirements*.txt`, CI | |

**Bàn giao:** web chạy được end-to-end + bảng ablation đầy đủ.

**Ước lượng:** nhiều việc nhỏ, ít phụ thuộc — làm song song được từ sớm bằng dữ liệu giả.

---

## 2. Việc chung (cả nhóm)

Không ai được đứng ngoài các việc sau:

| Việc | Ai |
|---|---|
| Viết đề cương, báo cáo | D chủ trì, cả nhóm viết phần của mình |
| Làm slide | C chủ trì, mỗi người 1–2 slide thuộc phần mình |
| Chạy ablation cuối kỳ | B + C |
| Kiểm tra số liệu trước khi báo cáo | người không viết phần đó |
| Review PR | luân phiên, mỗi PR cần 1 approve |

---

## 3. Ranh giới trách nhiệm

Quy tắc để tránh "ai cũng làm mà không ai chịu":

- **Một file có một chủ.** Người khác muốn sửa → mở PR, chủ file review.
- **Sửa `docs/` thì không cần xin phép** ai, nhưng phải ghi rõ trong commit.
- **Đổi interface giữa hai tầng** (ví dụ `search()` trả gì) → phải báo người kia **trước khi** code, không phải sau.
- **Đổi `configs/base.yaml`** → báo cả nhóm, vì nó ảnh hưởng mọi kết quả đã chạy.

---

## 4. Lịch

| Tuần | Mốc |
|---|---|
| 1 | Dựng repo, thống nhất interface, A chạy xong Doc Store |
| 2 | A xong 2 chỉ mục; B chạy được 2 nhánh trên chỉ mục thật |
| 3 | C xong RRF + rerank; D xong web khung + trec_eval |
| 4 | Chạy ablation đầy đủ; D hoàn thiện snippet |
| 5 | Viết báo cáo, làm slide, tập thuyết trình |

Chi tiết từng task: `docs/ROADMAP.md`.
