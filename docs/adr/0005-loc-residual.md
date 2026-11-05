# 0005: Postfilter residual là chế độ tùy chọn, mặc định tắt

- **Ngày:** 2026-11-05
- **Người:** Người 3 (Fusion · Rerank · Đánh giá)
- **Trạng thái:** chấp nhận

## Bối cảnh

TREC-COVID dùng residual collection: mỗi vòng chỉ chấm trên phần tài liệu chưa được chấm.
ADR 0002 và bản đầu của `docs/ARCHITECTURE.md` yêu cầu "lọc residual theo
`docids-rnd3.txt` trước khi ghi run". ADR 0008 đã đếm và kết luận `docids-rnd3.txt` là
**danh sách được phép** (trùng khít Doc Store), không phải danh sách bài đã chấm.
Vì thế "lọc residual" có hai nghĩa khác nhau, và phải đo cả hai trước khi chốt (TEAM.md,
ROADMAP Mốc 2.7).

`src/trec_covid/postfilter/residual.py` cài đặt ba chế độ:

| Chế độ | Làm gì | Nguồn |
|---|---|---|
| `off` | không lọc | — |
| `docids` | chỉ giữ bài nằm trong danh sách được phép | `docids-rnd3.txt` |
| `judged` | với mỗi topic, bỏ bài đã có phán định ở vòng trước (mọi nhãn, kể cả 0) | `qrels.csv` |

## Số đo (2026-11-05)

Run đầu vào: `convex (α = 0.3) + rerank top-50`, top-1000, dense **tạm**.
Bảng: `results/2026-11-05_residual.csv`.

| Chế độ | Dòng bị bỏ | nDCG@10 | nDCG@20 | MAP | P@5 | bpref |
|---|---|---|---|---|---|---|
| `off` | 0 | 0.2230 | 0.2114 | 0.0707 | 0.2343 | 0.3382 |
| `docids` | **0** | 0.2230 | 0.2114 | 0.0707 | 0.2343 | 0.3382 |
| `judged` | 3.950 | **0.0000** | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

- `docids`: không bài nào bị bỏ. Năm cột đầu của run file giống hệt `off`.
- `judged`: top-1000 của 35 topic chứa 3.950 bài có phán định, trong đó 2.045 bài liên
  quan. Chế độ này bỏ **toàn bộ** số đó, nên mọi độ đo bằng 0. Hệ thống không kém đi;
  thước đo đã mất hết nhãn. `qrels.csv` chỉ có phán định vòng 1–2, đúng tập mà residual
  vòng 3 phải loại.

## Phương án đã cân nhắc

| Phương án | Ưu | Nhược |
|---|---|---|
| Luôn bật `docids` | đúng thể lệ nộp bài | không đổi gì (no-op), tốn một lần đọc file |
| Luôn bật `judged` | đúng nghĩa residual khi nộp vòng 3 | chấm cục bộ ra 0; trên UI ẩn mất bài liên quan nhất |
| **Mặc định `off`, hai chế độ kia là tùy chọn** | đánh giá offline và UI không mất bài; vẫn có đường đúng thể lệ khi cần | phải nhớ bật khi tạo run nộp thi |

## Quyết định

- `configs/postfilter.yaml: mode: "off"` là mặc định cho đánh giá offline, bảng ablation
  và UI.
- `docids` giữ làm chế độ "đúng thể lệ" khi xuất run nộp vòng 3. Với Doc Store bản
  19/05/2020 nó không đổi điểm.
- `judged` chỉ dùng khi có qrels của vòng **sau** vòng đang nộp (ở đây là qrels vòng 3+,
  nhóm không có). **Không bật trên UI**: nó ẩn đúng những bài mà người dùng cần nhất.

## Hệ quả

- Dòng "Residual" trong bảng ablation 5 tầng ghi số của `docids` (bằng tầng trước) và
  ghi chú số của `judged` (0) kèm lý do. Không dùng nó để kết luận hệ thống tốt hay kém.
- `scripts/04_make_run.py --postfilter {off,docids,judged}` chọn chế độ. `--from-run`
  lọc một run có sẵn mà không phải rerank lại.
- YAML 1.1 đọc chữ `off` trần thành `False`, nên config phải ghi `"off"` trong ngoặc
  kép. Có test khoá lại.
- Nếu đổi sang bản CORD-19 khác thì `docids` có thể không còn là no-op. Khi đó phải đo lại.
