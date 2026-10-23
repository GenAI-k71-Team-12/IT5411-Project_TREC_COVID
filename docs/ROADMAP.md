# Kế hoạch triển khai

Thứ tự công việc bám theo **phụ thuộc dữ liệu**, không phải theo số thứ tự giai đoạn. Việc nào không cần dữ liệu thật thì làm trước bằng dữ liệu giả.

Ký hiệu người phụ trách: **A** · **B** · **C** · **D** (xem `docs/TEAM.md`).

---

## Mốc 0 — Khởi tạo (tuần 1, cả nhóm)

| # | Việc | Ai | Xong khi |
|---|---|---|---|
| 0.1 | Điền tên/MSSV vào `docs/TEAM.md` và bìa báo cáo | cả nhóm | không còn dấu `…` |
| 0.2 | Chốt interface giữa các tầng (xem dưới) | cả nhóm | ghi vào `docs/ARCHITECTURE.md` |
| 0.3 | Đặt dữ liệu vào `data/raw` và `data/external` | A | 4 file đúng vị trí |
| 0.4 | `pip install -r requirements.txt` chạy được trên cả 4 máy | D | 4/4 máy không lỗi |

### Interface cần chốt trước khi ai viết code

```python
# io
def load_doc_store() -> pd.DataFrame: ...        # cột: cord_uid, title, abstract, journal, year, doi

# retrieve
def search(query: str, top_k: int = 1000) -> list[tuple[str, float]]: ...

# fuse
def fuse(runs: dict[str, list[tuple[str, float]]]) -> list[tuple[str, float]]: ...

# rerank
def rerank(query: str, candidates: list[str]) -> list[tuple[str, float]]: ...

# postfilter
def filter_residual(ranked: list[tuple[str, float]]) -> list[tuple[str, float]]: ...

# snippet
def make_snippet(query: str, doc: dict) -> str | None: ...
```

**Lý do chốt sớm:** bốn người code bốn tầng; nếu interface đổi giữa chừng thì cả ba tầng sau phải sửa lại.

---

## Mốc 1 — Chỉ mục (tuần 1–2) — A

| # | Việc | Xong khi |
|---|---|---|
| 1.1 | Đọc `metadata.csv` → Doc Store | `data/processed/doc_store.parquet`, ~170k dòng, kiểm tra số dòng khớp |
| 1.2 | Đếm bài thiếu abstract | báo cáo con số, xác nhận vẫn giữ lại |
| 1.3 | Dựng chỉ mục Lucene (`title² + abstract`) | `indexes/lucene/`, nạp lại được |
| 1.4 | Hiệu chỉnh `k1`, `b` trên tập dev | ghi kết quả vào `results/`, chốt trong `configs/bm25.yaml` |
| 1.5 | Encode toàn bộ corpus bằng MedCPT Article Encoder | `indexes/faiss/`, ~2–3 giờ, cache xuống đĩa |
| 1.6 | Chuẩn hoá L2 + nạp `IndexFlatIP` | cùng một query, `IP` khớp `cosine` sai số < 1e-5 |
| 1.7 | Kiểm tra toàn vẹn chỉ mục | in SHA256, ghi vào `docs/` để người khác xác nhận |

**Cổng kiểm soát:** 10 truy vấn mẫu, mỗi nhánh trả đúng 1000 kết quả trong < 10 giây. Chưa qua cổng này thì B và C chưa bắt đầu được.

---

## Mốc 2 — Truy xuất (tuần 2) — B

| # | Việc | Xong khi |
|---|---|---|
| 2.1 | Hàm `search()` cho BM25 | chạy trên chỉ mục thật, < 500 ms |
| 2.2 | Hàm `search()` cho dense | encode query bằng **Query-Encoder** (không dùng Article) |
| 2.3 | Chuẩn hoá truy vấn dùng chung | hai nhánh dùng **cùng** một hàm, tránh lệch |
| 2.4 | Ghi chú các truy vấn mà hai nhánh bất đồng | ≥ 3 ví dụ, dùng cho slide "vì sao cần BM25" |

**Cổng kiểm soát:** chạy 35 topic, ghi ra 2 run file riêng (`runs/bm25.txt`, `runs/dense.txt`) và chấm bằng `trec_eval` — có số liệu baseline của từng nhánh.

---

## Mốc 3 — Fusion + rerank (tuần 3) — C

| # | Việc | Xong khi |
|---|---|---|
| 3.1 | RRF chuẩn, `k = 60` | vượt được cả hai baseline đơn nhánh |
| 3.2 | Weighted RRF, học `w` trên dev | tìm được `w` tốt nhất, ghi lại |
| 3.3 | Convex sau min-max, học `α` trên dev | min-max tính trong từng truy vấn |
| 3.4 | Cross-Encoder chấm top-50 | chạy batch, cắt 512 token |
| 3.5 | **Ghi điểm đơn điệu giảm** sau rerank | kiểm tra bằng script: `score[i] >= score[i+1]` |
| 3.6 | So sánh 3 công thức | bảng kết quả trong `results/` |

**Cổng kiểm soát:** bảng so sánh 3 công thức + 2 baseline, chênh lệch có ý nghĩa.

---

## Mốc 4 — Hậu xử lý & web (tuần 3–4) — D

Làm song song từ tuần 1 bằng dữ liệu giả, không chờ chỉ mục thật.

| # | Việc | Xong khi |
|---|---|---|
| 4.1 | Viết hàm tách câu + danh sách viết tắt | unit test với 5 abstract có `et al.`, `Fig.`, `0.5 mg.` |
| 4.2 | Chấm câu `(query, câu)` theo batch | 50 cặp < 1 giây trên máy nhóm |
| 4.3 | Tô sáng từ khoá | không tô sai chữ, không lệch vị trí |
| 4.4 | Fallback bài không có abstract | hiển thị metadata, không vỡ layout |
| 4.5 | Lọc residual theo `docids-rnd3.txt` | run file không chứa docid ngoài tập |
| 4.6 | Ghi run file 6 cột | định dạng `topicid Q0 docid rank score tag` |
| 4.7 | Wrapper `trec_eval` | `-m ndcg_cut.10`, `-m map`, `-m P.5`, `-m bpref` |
| 4.8 | Web Flask: trang kết quả + phân trang 10 | chạy `localhost:5000` |
| 4.9 | Cache snippet theo `(query, docid)` | lần 2 xem lại trang cũ: 0 ms |

**Cổng kiểm soát:** mở web, gõ 1 truy vấn, thấy 10 kết quả có snippet tô sáng + trích dẫn nguồn, bấm sang trang 2 vẫn đúng.

---

## Mốc 5 — Đánh giá đầy đủ (tuần 4) — B + C

| # | Việc | Xong khi |
|---|---|---|
| 5.1 | Bảng ablation: BM25 · Dense · RRF · +rerank · +residual | 5 dòng, cùng 35 topic |
| 5.2 | Kiểm tra 5 topic 36–40 bị loại khỏi trung bình | ghi rõ trong báo cáo "đánh giá trên 35 topic" |
| 5.3 | Chạy lại toàn bộ từ commit sạch | kết quả trùng khớp (tái lập 100%) |
| 5.4 | Ghi kèm commit hash vào mỗi file kết quả | `results/*.txt` có dòng đầu ghi hash + config |

---

## Mốc 6 — Báo cáo & thuyết trình (tuần 5)

| # | Việc | Ai |
|---|---|---|
| 6.1 | Chương phương pháp | A + B |
| 6.2 | Chương thực nghiệm & bảng kết quả | C |
| 6.3 | Chương sản phẩm & demo | D |
| 6.4 | Slide 7–10 phút, mỗi người 1–2 slide | cả nhóm |
| 6.5 | Tập thuyết trình, bấm giờ | cả nhóm |
| 6.6 | Chuẩn bị câu hỏi phản biện (xem dưới) | cả nhóm |

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
