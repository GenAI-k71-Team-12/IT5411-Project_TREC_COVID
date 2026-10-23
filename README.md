# TREC-COVID — Hệ thống Tìm kiếm Thông tin Y sinh (TVDL)

Đồ án môn **Truy vấn Dữ liệu lớn** — nhóm 4 sinh viên.

| | |
|---|---|
| **Đề tài** | Xây dựng hệ thống tìm kiếm thông tin y sinh quy mô lớn áp dụng Hybrid Search trên tập dữ liệu TREC-COVID |
| **Bộ dữ liệu** | CORD-19, bản phát hành 19/05/2020 (~170 nghìn bài báo) |
| **Cuộc thi** | TREC-COVID Round 3 (NIST), đóng 03/06/2020 |
| **Đơn vị truy xuất** | Bài báo (`cord_uid`) — không phải chunk |
| **Sản phẩm** | Trang web tìm kiếm (Flask), trả về xếp hạng kèm trích dẫn mức câu |

> **Tài liệu tham chiếu:** đề cương và slide nằm ở thư mục cha (`../Đề cương hiệu chỉnh.docx`, `../Thuyết trình_sửa.pptx`). Bản sao lưu trữ khi nộp nằm trong `docs/report/` và `docs/slides/`.

---

## 1. Kiến trúc pipeline

Mỗi truy vấn đi qua bốn giai đoạn. Chỉ giai đoạn 2–4 chạy lúc người dùng tìm kiếm; giai đoạn 1 chạy một lần rồi cache xuống đĩa.

```
CORD-19 (19/05/2020)
      │
 [GĐ1] Tiền xử lý & xây chỉ mục  (một lần, ~2–3 giờ)
      │      Doc Store (title + abstract)   ← giữ nguyên văn, KHÔNG lưu câu đã tách
      ├──► Lucene BM25 (fielded)            ← title² + abstract, EnglishAnalyzer
      └──► FAISS IndexFlatIP (MedCPT)       ← 768 chiều, chuẩn hoá L2 → IP = cos

Query
      │
 [GĐ2] Truy xuất đa nhánh      (< 1 giây)
      ├──► BM25   top-1000
      └──► Dense  top-1000
      │
 [GĐ3] Dung hợp + rerank
      │      RRF (mức bài báo) ──► Cross-Encoder top-50
      │
 [GĐ4] Hậu xử lý
      │      Lọc residual (docids-rnd3.txt)
      │      Tách câu abstract của 10 bài đầu ──► Snippet có tô sáng
      │
      ▼
  Trang kết quả (Flask)
```

Ba quyết định kiến trúc cốt lõi (xem `docs/adr/`):

1. **Đánh giá bằng `trec_eval` cục bộ**, không nộp Kaggle — chủ động chạy ablation nhiều lần.
2. **Tách câu lúc truy vấn**, không lưu `sentences[]` cho 170k bài — chỉ 10 bài/trang cần, ~10 ms.
3. **Snippet là trích nguyên văn** từ abstract — không tóm tắt, không dùng LLM sinh văn bản.

---

## 2. Cấu trúc thư mục

```
TREC_COVID_TVDL/
├─ configs/                  # Tham số từng tầng (YAML) — không hard-code trong code
├─ data/
│  ├─ raw/                   # CORD-19 gốc (KHÔNG commit, ~12.7 GB)
│  ├─ interim/               # Trung gian sau khi parse
│  ├─ processed/             # Doc Store đã chuẩn hoá (parquet/jsonl)
│  └─ external/              # topics-rnd3.csv, docids-rnd3.txt, qrels.csv
├─ indexes/
│  ├─ lucene/                # Chỉ mục BM25 (KHÔNG commit)
│  └─ faiss/                 # Chỉ mục vector (KHÔNG commit, ~500 MB)
├─ src/trec_covid/           # Mã nguồn chính (package Python)
│  ├─ config.py              # Đọc/merge YAML
│  ├─ cli.py                 # Entry point thống nhất
│  ├─ io/                    # Đọc CORD-19, Doc Store
│  ├─ preprocess/            # Chuẩn hoá văn bản, tách câu
│  ├─ index/                 # Dựng chỉ mục BM25 + FAISS
│  ├─ retrieve/              # Hai nhánh truy xuất
│  ├─ fuse/                  # RRF / Weighted RRF / Convex
│  ├─ rerank/                # Cross-Encoder
│  ├─ postfilter/            # Lọc residual, ghi run file
│  ├─ snippet/               # Trích dẫn mức câu
│  ├─ evaluate/              # trec_eval wrapper, ablation
│  └─ web/                   # Ứng dụng Flask
├─ scripts/                  # Lệnh chạy theo từng bước (01_ … 06_)
├─ notebooks/                # Thí nghiệm, phân tích (không phải code production)
├─ tests/                    # Unit test (pytest)
├─ runs/                     # Run file xuất ra để chấm (KHÔNG commit)
├─ results/                  # Kết quả trec_eval, bảng ablation
├─ docs/
│  ├─ ARCHITECTURE.md        # Chi tiết kỹ thuật từng giai đoạn
│  ├─ TEAM.md                # Phân công, quy ước làm việc
│  ├─ ROADMAP.md             # Mốc tiến độ
│  ├─ adr/                   # Architecture Decision Records
│  ├─ report/                # Đề cương, báo cáo cuối kỳ
│  ├─ slides/                # Slide thuyết trình
│  └─ diagrams/              # Sơ đồ nguồn (drawio, plantuml)
└─ .github/                  # Template PR / issue
```

---

## 3. Thư viện sử dụng

| Tầng | Thư viện | Ghi chú |
|---|---|---|
| BM25 | `pyserini` (Lucene) | cần **Java 11+**; phương án dự phòng `bm25s` |
| Vector | `faiss-cpu` | `IndexFlatIP`, không có bước train |
| Mô hình | `transformers`, `torch` | MedCPT: Query / Article / Cross — 3 checkpoint riêng |
| Dữ liệu | `pandas`, `pyarrow` | đọc `metadata.csv` |
| Web | `flask` | chạy cục bộ, không cần mạng |
| Đánh giá | `trec_eval` (binary) | tải riêng, không phải package Python |

> **Không dùng** `rank_bm25`: quét O(N) mỗi truy vấn, làm chậm mọi lần chạy ablation.

---

## 4. Bắt đầu

```powershell
git clone <repo-url>
cd TREC_COVID_TVDL
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt
copy .env.example .env
```

Sau đó đặt dữ liệu vào `data/`:

| File | Lấy từ | Đặt vào |
|---|---|---|
| CORD-19 (19/05/2020) | cuộc thi / AI2 | `data/raw/cord19-2020-05-19/` |
| `topics-rnd3.csv` | tab Data của cuộc thi | `data/external/` |
| `docids-rnd3.txt` | tab Data của cuộc thi | `data/external/` |
| `qrels.csv` | tab Data của cuộc thi | `data/external/` |

> Mã nguồn chưa được viết. Xem `docs/ROADMAP.md` để biết thứ tự triển khai.

---

## 5. Lệnh chạy (dự kiến)

```powershell
python scripts/01_build_docstore.py      # CORD-19 → Doc Store
python scripts/02_build_bm25_index.py    # Doc Store → Lucene
python scripts/03_build_faiss_index.py   # Doc Store → FAISS (~2–3 giờ)
python scripts/04_make_run.py --config configs/fusion.yaml
python scripts/05_evaluate.py --run runs/rrf.txt
python scripts/06_serve_web.py           # http://localhost:5000
```

---

## 6. Nhóm

| Thành viên | Mảng phụ trách | Giai đoạn |
|---|---|---|
| … | Dữ liệu & chỉ mục | GĐ1 |
| … | Truy xuất hai nhánh | GĐ2 |
| … | Dung hợp & rerank | GĐ3 |
| … | Trích dẫn, đánh giá & web | GĐ4 |

Chi tiết phân công và quy ước nhánh/commit: `docs/TEAM.md` và `CONTRIBUTING.md`.
