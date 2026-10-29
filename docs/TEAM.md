# Phân công nhóm

Điền tên thật vào bảng dưới khi bắt đầu làm.

| Mã | Họ tên | MSSV | Vai trò | Trách nhiệm chính |
|---|---|---|---|---|
| **Người 1** | ……………… | ………… | Dữ liệu & Tìm kiếm BM25 | Doc Store, chuẩn hoá văn bản, chỉ mục Lucene, truy xuất BM25 |
| **Người 2** | ……………… | ………… | Tìm kiếm Dense Vector | Encode MedCPT, chỉ mục FAISS, truy xuất Dense |
| **Người 3** | ……………… | ………… | Dung hợp · Rerank · Đánh giá | Fusion, Cross-Encoder, postfilter, `trec_eval`, ablation |
| **Người 4** | ……………… | ………… | Snippet & Web UI | Tách câu y sinh, tô sáng, cache, toàn bộ giao diện Flask |

Nhóm trưởng: ……………… (luân phiên theo tuần nếu cần)

Giảng viên hướng dẫn: ………………

---

## 1. Phạm vi từng người

Phân công được chia theo **phương pháp truy xuất**, không theo giai đoạn: mỗi thành viên sở hữu một **nhánh dọc hoàn chỉnh** — tự dựng chỉ mục, tự truy vấn, tự chấm điểm bằng `trec_eval`. Cách chia này cho phép mỗi người làm việc độc lập ngay từ tuần đầu nhờ dữ liệu giả lập, và phối hợp qua chuẩn [`docs/DATA_CONTRACTS.md`](DATA_CONTRACTS.md).

| Người | Sở hữu | LOC ước tính | Compute | Bị chặn bởi |
|---|---|---|---|---|
| **Người 1** | Doc Store + preprocess + BM25 index + `search_bm25` + tuning `k1`,`b` | ~450 | ~15 phút | — |
| **Người 2** | Encode 170k + FAISS + `search_dense` | ~300 | **2–3 giờ GPU** | Doc Store tối giản (Người 1) |
| **Người 3** | Fusion + rerank + postfilter + run file + `evaluate/` | ~600 | 15–30 phút | Người 1 & 2 |
| **Người 4** | Snippet + **toàn bộ Web UI** | ~700 | không đáng kể | — (dùng mock data) |

> **Nguyên tắc:** mỗi file có đúng một chủ. Không có việc nào ghi "cả nhóm cùng code X" — mọi đóng góp chéo đều qua Pull Request.

### Người 1 — Dữ liệu & Tìm kiếm BM25 (Lexical / Sparse Retrieval)

| Sở hữu | File / thư mục |
|---|---|
| **Doc Store:** `metadata.csv` → parquet, chuẩn hoá, giữ `abstract` nguyên chuỗi | `src/trec_covid/io/doc_store.py`, `scripts/01_build_docstore.py` |
| Bảng ánh xạ `position → cord_uid` cho FAISS | `data/processed/docid_order.parquet` |
| Chuẩn hoá an toàn dùng chung cho document và query; `escape_lucene()` riêng, chỉ bật khi dùng raw query parser | `src/trec_covid/preprocess/text.py` |
| Xây dựng chỉ mục BM25 một trường (`title` lặp ×2 + `abstract`, xấp xỉ `title² + abstract`) | `src/trec_covid/index/bm25_index.py` |
| Đọc topic cuộc thi (`query` / `question` / `narrative`) | `src/trec_covid/io/topics.py` |
| Truy vấn BM25 top-1000 | `src/trec_covid/retrieve/bm25.py` |
| Tinh chỉnh siêu tham số `k1`, `b` trên tập dev | `configs/bm25.yaml` |
| Script dựng chỉ mục BM25 | `scripts/02_build_bm25_index.py` |

- **Mốc chặn:** **Doc Store tối giản** (`cord_uid`, `title`, `abstract`) giao trong **2 ngày đầu** — Người 2 cần nó để bắt đầu encode. Bổ sung 5 trường metadata còn lại sau, không phải encode lại.
- **Bàn giao:** Hàm `search_bm25(query, top_k=1000) -> list[ScoredDocument]` chạy < 500 ms trên chỉ mục Lucene (hoặc `bm25s` dự phòng), kèm run file BM25 và `nDCG@10` baseline.
- **Quy ước:** Tuân thủ `ScoredDocument` theo [`src/trec_covid/schemas.py`](../src/trec_covid/schemas.py).

### Người 2 — Xây dựng tìm kiếm Dense Vector (Semantic / Embedding Retrieval)

| Sở hữu | File / thư mục |
|---|---|
| Tích hợp mô hình MedCPT (Query & Article Encoder) | `src/trec_covid/index/dense_index.py` |
| Encode toàn bộ corpus (~170k bài) thành vector 768 chiều | `indexes/faiss/` |
| Chuẩn hoá L2 và nạp chỉ mục FAISS `IndexFlatIP` | `src/trec_covid/index/dense_index.py` |
| Truy vấn Dense top-1000 bằng Query Encoder | `src/trec_covid/retrieve/dense.py` |
| Quản lý cấu hình & batching GPU/CPU | `configs/dense.yaml`, `scripts/03_build_faiss_index.py` |

- **Bàn giao:** Hàm `search_dense(query, top_k=1000) -> list[ScoredDocument]` chạy < 500 ms, chỉ mục vector nạp được trong < 10 giây.
- **Quy ước:** Tuân thủ `ScoredDocument` theo `src/trec_covid/schemas.py`.

### Người 3 — Dung hợp · Rerank · Đánh giá (Fusion, Re-ranking & Evaluation)

| Sở hữu | File / thư mục |
|---|---|
| Ba công thức dung hợp: RRF, Weighted RRF, Convex min-max | `src/trec_covid/fuse/` |
| Tái xếp hạng bằng MedCPT Cross-Encoder cho top-50 bài | `src/trec_covid/rerank/` |
| Postfilter: lọc residual (**tùy chọn**) + assert bất biến (điểm giảm dần, không trùng) | `src/trec_covid/postfilter/` |
| Wrapper `trec_eval` (`-m ndcg_cut.10`), ghi run file 6 cột, bảng ablation 5 tầng | `src/trec_covid/evaluate/` |
| Cấu hình dung hợp và rerank | `configs/fusion.yaml`, `configs/rerank.yaml` |
| Script tạo run file & chấm điểm | `scripts/04_make_run.py`, `scripts/05_evaluate.py` |

- **Mốc chặn:** **Bảng ablation 5 tầng** cuối tuần 4 — trễ là không kịp viết chương thực nghiệm.
- **Bàn giao:** Hàm `fuse_and_rerank(query, bm25_results, dense_results) -> list[ScoredDocument]`, đảm bảo điểm số đơn điệu giảm, kèm bảng so sánh 3 công thức fusion và `nDCG@10` vượt từng nhánh đơn lẻ.
- **Quy ước:** Nhận đầu vào và trả đầu ra theo đúng [`docs/DATA_CONTRACTS.md`](DATA_CONTRACTS.md).
- **Quyết định cần đo:** lọc residual **có thể làm giảm** `nDCG@10` khi chấm cục bộ trên chính tập qrels. Phải chạy cả hai cách rồi mới chốt ADR 0005.

### Người 4 — Snippet & Web UI (Evidence Extraction & Product)

| Sở hữu | File / thư mục |
|---|---|
| Tách câu chuyên sâu cho abstract y sinh (xử lý `et al.`, `Fig.`, `i.e.`, `0.5 mg.`) | `src/trec_covid/snippet/splitter.py` |
| Chấm điểm & chọn câu dẫn chứng, tô sáng `<mark>`, cache theo `(query_hash, cord_uid)` | `src/trec_covid/snippet/scorer.py` |
| Cấu hình snippet & xử lý bài không có abstract | `configs/snippet.yaml` |
| **Toàn bộ ứng dụng Web:** Flask, routing, template Jinja2, phân trang 10 bài/trang | `src/trec_covid/web/` |
| CSS/JS/font **cục bộ — cấm CDN** | `src/trec_covid/web/static/` |
| Script chạy web | `scripts/06_serve_web.py` |

- **Mốc chặn:** **Khung web chạy được với mock data** cuối tuần 2 — demo sớm để không phát hiện lỗi tích hợp vào tuần cuối.
- **Bàn giao:** Hàm `generate_snippets(query, documents) -> dict[str, SnippetResult]` xử lý 10 bài < 50 ms, trích nguyên văn câu, tô sáng chuẩn xác; giao diện tìm kiếm chạy cục bộ tại `localhost:5000`.
- **Quy ước:** Tuân thủ `SnippetResult` theo [`src/trec_covid/schemas.py`](../src/trec_covid/schemas.py).

---

## 2. Đặc tả UI & Việc chung

### 2.1. Đặc tả giao diện (Người 4 thực hiện)

- **Công nghệ:** Flask + HTML5/CSS3/JavaScript thuần (responsive, không dùng CDN — chạy được offline).
- **Trang chủ:** thanh tìm kiếm trung tâm, có trạng thái loading / empty / error.
- **Thẻ kết quả:** tiêu đề, tác giả, tạp chí, `publish_time`, liên kết DOI, điểm số, snippet đã tô sáng.
- **Phân trang:** 10 bài/trang, giữ đúng state khi chuyển trang.
- **Chế độ so sánh:** hiển thị điểm BM25 / Dense / Fusion / sau Rerank cho cùng một truy vấn (phục vụ demo).

### 2.2. Việc chung thật sự (không còn "cùng code")

| Việc chung | Ai chủ trì |
|---|---|
| Review PR | luân phiên — mỗi PR cần 1 người approve |
| Viết báo cáo | mỗi người 1 chương: P1 dữ liệu & BM25 · P2 dense · P3 thực nghiệm · P4 sản phẩm |
| Làm slide | mỗi người 1–2 slide thuộc phần mình |
| Chạy lại ablation từ commit sạch | Người 3 + 1 người review chéo |
| Điền tên/MSSV vào bảng đầu trang | cả nhóm, tuần 1 |
| Tập thuyết trình, bấm giờ | cả nhóm, tuần 5 |

### 2.3. Ba mốc chặn — trễ là trễ cả nhóm

| Mốc | Ai | Hạn | Nếu trễ thì sao |
|---|---|---|---|
| **Doc Store tối giản** (3 trường) | Người 1 | **Ngày 2** | Người 2 không encode được → mất cả tuần |
| **Khung web chạy với mock data** | Người 4 | **Cuối tuần 2** | Tuần 4 mới phát hiện lỗi tích hợp |
| **Bảng ablation 5 tầng** | Người 3 | **Cuối tuần 4** | Không kịp viết chương thực nghiệm |

---

## 3. Ranh giới trách nhiệm & Quy ước làm việc

- **Hợp đồng dữ liệu là tối thượng:** Không tự ý thay đổi tên trường, kiểu dữ liệu trong [`src/trec_covid/schemas.py`](../src/trec_covid/schemas.py) và [`docs/DATA_CONTRACTS.md`](DATA_CONTRACTS.md) nếu chưa thảo luận với cả nhóm.
- **Một file có một chủ:** Ai phụ trách mô-đun nào thì chủ trì file đó. Mọi đóng góp khác phải thông qua Pull Request (PR).
- **Lập trình độc lập nhờ Mock Data:** Tận dụng dữ liệu giả lập trong `DATA_CONTRACTS.md` để viết unit test ngay từ ngày đầu, không cần chờ đợi người khác code xong.
- **Review chéo PR:** Mỗi PR cần ít nhất 1 thành viên khác review và phê duyệt trước khi merge vào nhánh tích hợp `dev`.

---

## 4. Lịch trình dự kiến

| Tuần | Người 1 (Dữ liệu & BM25) | Người 2 (Dense) | Người 3 (Fusion · Rerank · Đánh giá) | Người 4 (Snippet & UI) |
|---|---|---|---|---|
| **Tuần 1** | ⏰ Doc Store tối giản (ngày 2) → Doc Store đầy đủ → dựng Lucene | ⏰ **Bắt đầu encode ngay ngày 1** | 3 công thức fusion trên mock data | Tách câu + tô sáng trên mock docs |
| **Tuần 2** | `search_bm25()` + tuning `k1`,`b` | Encode xong → `search_dense()` + kiểm chứng IP ≈ cosine | Nối fusion với kết quả thật | ⏰ **Khung web chạy với mock** |
| **Tuần 3** | Hỗ trợ tích hợp, viết chương phương pháp | Tối ưu tốc độ, viết chương dense | Rerank top-50 + run file | Nối pipeline thật, phân trang, cache |
| **Tuần 4** | Viết báo cáo phần mình | Viết báo cáo phần mình | ⏰ **Ablation 5 tầng** + ADR residual | Chế độ so sánh trên UI |
| **Tuần 5** | Slide + tập thuyết trình | Slide + tập thuyết trình | Slide + chạy lại từ commit sạch | Slide + demo |
