# 0002 — Đánh giá bằng trec_eval cục bộ, không nộp Kaggle

- **Trạng thái:** chấp nhận

## Bối cảnh

Cuộc thi có kênh nộp bài trên Kaggle. Nhưng mỗi ngày chỉ nộp được vài lần, và
mỗi lần chỉ trả về **một con số duy nhất** — không cho biết tầng nào đóng góp
bao nhiêu.

## Phương án đã cân nhắc

| Phương án | Ưu | Nhược |
|---|---|---|
| Nộp Kaggle | so sánh được với đội khác | giới hạn số lần; không ablation được |
| `trec_eval` cục bộ | chạy không giới hạn; ablation từng tầng; chủ động | phải tự dựng quy trình cho đúng |

## Quyết định

Dùng `trec_eval` cục bộ làm kênh đánh giá **duy nhất**. Không phụ thuộc Kaggle.

## Hệ quả

- Phải tự viết run file đúng định dạng 6 cột.
- Phải tự xử lý ba cái bẫy:
  - `trec_eval` sắp xếp lại theo cột `score` và bỏ qua cột `rank`
    -> điểm phải đơn điệu giảm theo thứ hạng cuối
  - mặc định không tính nDCG -> phải `-m ndcg_cut.10`
  - không dùng `-c` khi qrels chỉ có 35 topic
- Phải lọc residual theo `docids-rnd3.txt` trước khi ghi run.
- Không so sánh trực tiếp được với điểm trên bảng xếp hạng Kaggle nếu dùng
  tập topic khác (35 so với 40).
