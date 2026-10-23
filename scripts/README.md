# Script chạy

Mỗi script là một bước trong pipeline, đánh số theo **thứ tự chạy**, không
theo số giai đoạn trong slide.

| Script | Việc | Ghi chú |
|---|---|---|
| `01_build_docstore.py` | CORD-19 -> Doc Store | ~5 phút |
| `02_build_bm25_index.py` | Doc Store -> Lucene | ~10 phút |
| `03_build_faiss_index.py` | Doc Store -> FAISS | **~2-3 giờ** (encode) |
| `04_make_run.py` | Truy xuất + fusion + rerank -> run file | |
| `05_evaluate.py` | run file + qrels -> điểm | cần binary `trec_eval` |
| `06_serve_web.py` | chạy web | |

Chưa cài đặt. Tất cả script sẽ đọc tham số từ `configs/`, không nhận
đường dẫn tuyệt đối.
