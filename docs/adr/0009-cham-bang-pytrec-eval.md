# 0009: Chấm bằng `pytrec_eval-terrier` thay cho binary `trec_eval`

- **Ngày:** 2026-11-05
- **Người:** Người 3 (Fusion · Rerank · Đánh giá)
- **Trạng thái:** chấp nhận

## Bối cảnh

ADR 0002 chọn chấm cục bộ bằng `trec_eval`. Nhóm làm trên Windows, mà `trec_eval` là
chương trình C không có bản dựng sẵn cho Windows: mỗi máy phải tự cài `make` và `gcc`
rồi build. Cùng lúc đó, bảng ablation và việc tuning fusion cần chấm hàng trăm lần
trong một tiến trình Python.

## Phương án đã cân nhắc

| Phương án | Ưu | Nhược |
|---|---|---|
| Binary `trec_eval` qua `subprocess` | đúng nguyên văn ADR 0002 | build C trên 4 máy Windows; mỗi lần chấm phải ghi file tạm rồi parse stdout |
| `pytrec_eval-terrier` | bọc **đúng mã C** của trec_eval nên ra cùng con số; có wheel Windows/Python 3.13; chấm trực tiếp trên dict trong bộ nhớ | thêm một dependency pip |
| Tự viết nDCG/MAP bằng Python | không phụ thuộc gì | dễ lệch định nghĩa của trec_eval (gain, phá hoà, bpref) |

## Quyết định

Dùng `pytrec_eval-terrier` (`src/trec_covid/evaluate/trec_eval.py`). Không cần binary.

## Hệ quả

- Ba bẫy của ADR 0002 vẫn giữ nguyên, nay nằm trong code:
  - yêu cầu tường minh `ndcg_cut.10,20` (trec_eval mặc định không tính nDCG)
  - chỉ lấy trung bình trên topic có cả trong run lẫn qrels (tương đương không dùng
    `-c`), nên topic 36–40 không kéo điểm về 0
  - run file vẫn phải đơn điệu giảm theo `score`, vì trec_eval sắp lại theo `score`
    (hoà điểm thì docid giảm dần) và bỏ qua cột `rank`
- nDCG của trec_eval dùng **gain tuyến tính** (gain = nhãn 0/1/2), không phải
  `2^rel − 1`. Báo cáo phải ghi đúng công thức này. Có test khoá lại.
- Muốn đối chiếu với binary thì chạy `trec_eval -m ndcg_cut.10 qrels.txt run.txt` trên
  một máy có binary. Số phải khớp tới 4 chữ số.
- Nếu sau này cần độ đo mà pytrec_eval không có (ví dụ judged@k của bản trec_eval
  mới), tính thẳng bằng Python trong `evaluate/` và ghi rõ định nghĩa.
