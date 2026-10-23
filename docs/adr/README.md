# Architecture Decision Records

Mỗi quyết định kiến trúc ghi thành một file, **không bao giờ xoá** — chỉ thêm
ADR mới thay thế ADR cũ.

| # | Tiêu đề | Trạng thái |
|---|---|---|
| [0001](0001-mau-ghi-quyet-dinh.md) | Mẫu ghi quyết định | mẫu |
| [0002](0002-danh-gia-bang-trec-eval-cuc-bo.md) | Đánh giá bằng trec_eval cục bộ, không nộp Kaggle | chấp nhận |
| [0003](0003-tach-cau-luc-truy-van.md) | Tách câu lúc truy vấn, không lưu sentences[] | chấp nhận |
| [0004](0004-snippet-trich-nguyen-van.md) | Snippet là trích nguyên văn, không sinh văn bản | chấp nhận |

## Khi nào cần ADR

- Đổi interface giữa hai tầng
- Chọn thư viện / mô hình
- Thay đổi cách đánh giá
- Một phương án bị loại vì lý do không hiển nhiên
