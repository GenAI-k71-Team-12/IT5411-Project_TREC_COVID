# Kế hoạch triển khai

Thứ tự công việc bám theo **phụ thuộc dữ liệu** và **hợp đồng giao tiếp**. Mỗi người có thể làm việc độc lập ngay từ tuần đầu tiên bằng cách sử dụng dữ liệu giả lập (mock data) theo chuẩn [`docs/DATA_CONTRACTS.md`](docs/DATA_CONTRACTS.md).

Ký hiệu người phụ trách: **Người 1** (Dữ liệu & BM25) · **Người 2** (Dense) · **Người 3** (Fusion · Rerank · Đánh giá) · **Người 4** (Snippet & Web UI).

---

## Mốc 0 — Khởi tạo & Chốt quy ước (tuần 1, Cả nhóm)

| # | Việc | Ai | Xong khi |
|---|---|---|---|
| 0.1 | Điền tên/MSSV vào `docs/TEAM.md` và bìa báo cáo | Cả nhóm | không còn dấu `…` |
| 0.2 | Chốt Data Contracts & schemas | Cả nhóm | Thống nhất [`docs/DATA_CONTRACTS.md`](DATA_CONTRACTS.md) & [`src/trec_covid/schemas.py`](../src/trec_covid/schemas.py) |
| 0.3 | Đặt dữ liệu vào `data/raw` và `data/external` | Người 1 | 4 file đúng vị trí |
| 0.4 | `pip install -r requirements.txt` chạy được trên cả 4 máy | Cả nhóm | 4/4 máy không lỗi |
| 0.5 | **Doc Store tối giản** (`cord_uid`, `title`, `abstract`) — mốc chặn ngày 2 | **Người 1** | `data/processed/doc_store.parquet` đọc được, ~170k dòng |
| 0.6 | **Doc Store đầy đủ** 8 trường + `docid_order.parquet` | **Người 1** | Đủ trường, `len(docids) == index.ntotal` |
| 0.7 | Kiểm chứng `docids-rnd3.txt` là danh sách *được phép* hay *bị loại* | Người 3 | Ghi kết luận vào `docs/adr/` |

---

## Mốc 1 — Xây dựng hai nhánh tìm kiếm độc lập (tuần 1–2)

### Nhánh Lexical — Người 1 (BM25)
| # | Việc | Xong khi |
|---|---|---|
| 1.1 | Chuẩn hoá an toàn dùng chung cho document **và** query (một hàm duy nhất); stemming/stopword chỉ ở analyzer; `escape_lucene()` là hàm riêng, không gọi mặc định | Unit test: NBSP, gạch nối Unicode, HTML entity, query có `(` `)` `"` — xem ADR [0007](adr/0007-mot-lan-chuan-hoa-dung-chung.md) |
| 1.2 | Dựng chỉ mục BM25 **một trường** `title` lặp 2 lần + `abstract` (xấp xỉ `title² + abstract`) qua `pyserini`, dự phòng `bm25s` | `indexes/lucene/` + `meta.json`, nạp lại được — xem ADR [0006](adr/0006-mot-truong-van-ban-lap-title.md) |
| 1.3 | Viết hàm `search_bm25(query, top_k=1000) -> list[ScoredDocument]` | Trả đúng schema, latency < 500 ms |
| 1.4 | Hiệu chỉnh `k1`, `b` trên tập dev | Ghi kết quả vào `results/`, chốt trong `configs/bm25.yaml` |

### Nhánh Semantic — Người 2 (Dense Vector)
| # | Việc | Xong khi |
|---|---|---|
| 1.5 | Tích hợp checkpoint MedCPT Article Encoder & Query Encoder | Model nạp thành công trên GPU/CPU |
| 1.6 | Encode toàn bộ corpus (~170k bài) thành vector 768 chiều | Chạy batching, lưu `indexes/faiss/` |
| 1.7 | Chuẩn hoá L2 vector bài báo và nạp vào `IndexFlatIP` | Tích vô hướng khớp Cosine similarity |
| 1.8 | Viết hàm `search_dense(query, top_k=1000) -> list[ScoredDocument]` | Trả đúng schema, latency < 500 ms |

**Cổng kiểm soát:** Cả 2 nhánh chạy thử độc lập trên 35 topics, xuất run file baseline `runs/bm25.txt` và `runs/dense.txt` để đánh giá sơ bộ.

---

## Mốc 2 — Dung hợp & Reranker (tuần 2–3) — Người 3

Làm việc với Mock Data từ tuần 1, sau đó nối với kết quả thật từ Người 1 và Người 2.

| # | Việc | Xong khi |
|---|---|---|
| 2.1 | Cài đặt RRF chuẩn (`k = 60`) trên 2 danh sách `ScoredDocument` | Vượt được cả hai baseline đơn nhánh |
| 2.2 | Cài đặt Weighted RRF (tối ưu trọng số $w$ trên tập dev) | Tìm được $w$ tốt nhất, lưu config |
| 2.3 | Cài đặt Convex min-max kết hợp điểm số | Min-max chuẩn hóa theo từng truy vấn |
| 2.4 | Tích hợp MedCPT Cross-Encoder chấm top-50 bài sau dung hợp | Batch rerank chạy mượt mà |
| 2.5 | Đảm bảo tính đơn điệu giảm của điểm số sau rerank | `score[i] >= score[i+1]` cho toàn bộ top-1000 |
| 2.6 | Lọc residual theo `docids-rnd3.txt` (**chế độ tùy chọn**) + assert bất biến | Định dạng đúng 6 cột TREC; điểm giảm dần, không trùng docid |
| 2.7 | Thí nghiệm **có / không** lọc residual, so `nDCG@10` | Chốt ADR 0005 kèm số liệu hai chiều |

**Cổng kiểm soát:** Bảng so sánh 3 công thức fusion + rerank, cải thiện vượt trội so với baseline.

---

## Mốc 3 — Trích đoạn Snippet & Highlight (tuần 2–3) — Người 4

Làm việc với Mock Documents từ tuần 1.

| # | Việc | Xong khi |
|---|---|---|
| 3.1 | Thuật toán tách câu y sinh (regex nhận diện `et al.`, `Fig.`, `i.e.`, `0.5 mg.`) | Test case 5 abstract phức tạp không bị cắt cụt |
| 3.2 | Thuật toán chấm điểm câu liên quan nhất với query | Chọn được câu dẫn chứng tiêu biểu |
| 3.3 | Thuật toán tô sáng keyword (`<mark class="hl">...</mark>`) | Không làm vỡ định dạng HTML, khớp chính xác |
| 3.4 | Xử lý bài không có abstract (fallback) | Hiển thị thông báo hoặc trích tiêu đề không vỡ UI |
| 3.5 | Cơ chế cache snippet theo `(query, cord_uid)` | Lần 2 xem lại trang kết quả mất 0 ms |

**Cổng kiểm soát:** Hàm `generate_snippets` trả về đúng `dict[str, SnippetResult]` cho 10 bài < 50 ms.

---

## Mốc 4 — Xây dựng Web UI & Tích hợp Pipeline (tuần 2–4) — Người 4

| # | Việc | Ai | Xong khi |
|---|---|---|---|
| 4.1 | Dựng khung ứng dụng Flask, routing, template Jinja2 — **mốc chặn cuối tuần 2, chạy bằng mock data** | Người 4 | `localhost:5000` chạy mượt với dữ liệu giả lập |
| 4.2 | Thiết kế giao diện (search bar, responsive cards, metadata, snippet) | Người 4 | UI chuyên nghiệp, trực quan |
| 4.3 | Tích hợp phân trang (10 bài/trang) và bộ lọc năm/journal | Người 4 | Chuyển trang giữ đúng state và query |
| 4.4 | Nối toàn bộ pipeline: Query → BM25 + Dense → Fusion/Rerank → Snippet → UI | Người 4 | Tìm kiếm một từ khoá ra đầy đủ card + snippet tô sáng |
| 4.5 | Chế độ so sánh (Debug Mode): điểm BM25 vs Dense vs Cross-Encoder trên UI | Người 4 | Hỗ trợ demo trực tiếp cho thầy cô |
| 4.6 | CSS/JS/font cục bộ — **cấm CDN**, chạy được khi tắt mạng | Người 4 | Ngắt mạng vẫn render đúng |

---

## Mốc 5 — Đánh giá toàn diện & Ablation (tuần 4) — Người 3

| # | Việc | Ai | Xong khi |
|---|---|---|---|
| 5.1 | Chạy `trec_eval` tính `nDCG@10`, `nDCG@20`, `MAP`, `P@5`, `bpref` trên 35 topics | Người 3 | Đầy đủ file log kết quả |
| 5.2 | Lập bảng Ablation 5 tầng (BM25 → Dense → RRF → Rerank → Residual) — **mốc chặn cuối tuần 4** | Người 3 | Thể hiện rõ đóng góp của từng tầng |
| 5.3 | Kiểm tra tính tái lập (Replicability) từ commit sạch | Người 3 | Kết quả trùng khớp 100% |
| 5.4 | Ghi ADR 0005: có/không lọc residual, kèm số liệu hai chiều | Người 3 | ADR được merge vào `docs/adr/` |

> **Bẫy khi chấm:** `trec_eval` mặc định **không** tính nDCG (phải truyền `-m ndcg_cut.10`); **không** dùng cờ `-c` khi qrels chỉ có 35/40 topic.

---

## Mốc 6 — Báo cáo & Thuyết trình (tuần 5) — Cả nhóm

| # | Việc | Ai phụ trách chính |
|---|---|---|
| 6.1 | Viết chương Phương pháp (BM25, Dense, RRF, Cross-Encoder, Snippet) | Người 1, 2, 3, 4 viết phần của mình |
| 6.2 | Viết chương Thực nghiệm, bảng số liệu Ablation | Người 3 chủ trì |
| 6.3 | Viết chương Sản phẩm ứng dụng Web & Trích dẫn | Người 4 |
| 6.4 | Thiết kế slide thuyết trình (mỗi người 2-3 slide phần mình) | Cả nhóm |
| 6.5 | Tổng duyệt thuyết trình, chuẩn bị demo live và câu hỏi phản biện | Cả nhóm |

### Câu hỏi phản biện cần chuẩn bị sẵn

1. Đã có dense và cross-encoder, vì sao vẫn cần BM25?
2. Vì sao tách câu lúc truy vấn mà không tách sẵn khi dựng chỉ mục?
3. Snippet có phải do mô hình sinh ra không? Có nguy cơ bịa không?
4. nDCG@10 = 0,8 nghĩa là gì? Vì sao không dùng Accuracy?
5. Vì sao chỉ 35 topic được chấm, 5 topic còn lại thì sao?
6. Cross-encoder chấm 50 cặp mất bao lâu? Có làm chậm người dùng không?
7. Vì sao dùng `IndexFlatIP` mà không dùng `IndexIVFFlat`?
8. Kết quả có tái lập được không? Làm sao chứng minh?

---

## Rủi ro và phương án

| Rủi ro | Khả năng | Phương án |
|---|---|---|
| Encode 170k bài quá 3 giờ / hết RAM | trung bình | chia batch, ghi từng phần; máy nhóm chạy qua đêm |
| `pyserini` lỗi Java trên Windows | cao | dùng `bm25s` (thuần Python), chấp nhận chậm hơn |
| Cross-Encoder quá chậm trên CPU | cao | giảm còn top-30; hoặc dùng mô hình nhỏ hơn cho rerank |
| Chỉ mục quá lớn, không chia sẻ được | trung bình | nén, hoặc mỗi người tự dựng từ Doc Store (tất định) |
| Máy mạnh nhất hỏng | thấp | Doc Store + chỉ mục nằm trên ổ ngoài của nhóm, không chỉ trên 1 máy |
