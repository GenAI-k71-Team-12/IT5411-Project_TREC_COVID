# Script chạy

Mỗi script là một bước trong pipeline, đánh số theo **thứ tự chạy**, không
theo số giai đoạn trong slide.

| Script | Việc | Ghi chú |
|---|---|---|
| `01_build_docstore.py` | CORD-19 -> Doc Store | ~5 phút; `--minimal` = đợt bàn giao 1 |
| `02_build_bm25_index.py` | Doc Store -> Lucene BM25 | ~10 phút; `--backend bm25s` nếu không có Java |
| `03_build_faiss_index.py` | Doc Store -> FAISS | **~2-3 giờ** (encode) |
| `04_make_run.py` | Truy xuất + fusion + rerank -> run file | |
| `05_evaluate.py` | run file + qrels -> điểm | `pytrec_eval` (ADR 0009) |
| `06_serve_web.py` | chạy web | |
| `07_fusion_experiments.py` | 2 run nhánh -> lưới w/α, CV 5-fold -> bảng so sánh fusion | ~5 giây; chọn theo nDCG@10 condensed (ADR 0010) |

**Đã chạy được:** `01`, `02` (nhánh dữ liệu + BM25, Người 1); `05`, `07` (đánh giá và
fusion, Người 3). Các script còn lại mới ở mức khung (chưa có mã).

Tất cả script đọc tham số từ `configs/` (`--config bm25`), không hard-code
đường dẫn tuyệt đối; `--out`, `--doc-store`, `--metadata` chỉ để ghi đè khi
test.

Ví dụ chạy thử trên một CSV nhỏ (không cần dữ liệu thật):

```powershell
python scripts/01_build_docstore.py --metadata data/interim/mini.csv --out data/interim/mini.parquet --nrows 50 --minimal
python scripts/02_build_bm25_index.py --doc-store data/interim/mini.parquet --out data/interim/mini-index --backend bm25s
```
