# 0006 — Một trường văn bản, lặp title để xấp xỉ `title² + abstract`

- **Ngày:** 2026-10-29
- **Người:** Người 1 (Dữ liệu & BM25)
- **Trạng thái:** chấp nhận

## Bối cảnh

README chốt cách đánh chỉ mục là **`title² + abstract`**: tiêu đề mang tín hiệu
rất mạnh ở TREC-COVID (tên thuốc, tên bệnh, mã thử nghiệm), nên được tính trọng
số gấp đôi. Có ba cách hiện thực:

1. **Fielded BM25** — Lucene đánh chỉ mục hai trường `title` và `abstract`,
   dùng `BM25Similarity` với `setMultiValueBoost`/`MultifieldSimilarity` hoặc
   `BoostQuery` khi truy vấn.
2. **Một trường, lặp title** — ghép `title title abstract` thành một chuỗi,
   đánh chỉ mục một trường duy nhất.
3. **Hai lần chấm điểm** — chạy BM25 riêng trên title và abstract rồi cộng
   có trọng số.

Ràng buộc: cả hai backend phải cho **cùng một ngữ nghĩa**, vì `bm25s` (phương án
dự phòng khi máy không có Java) không hỗ trợ fielded BM25, và nhánh dense của
Người 2 cũng chỉ có **một** vector cho mỗi bài (không tách title/abstract).

## Phương án đã cân nhắc

| Phương án | Ưu | Nhược |
|---|---|---|
| A. Fielded BM25 (Lucene 2 trường) | đúng nghĩa trọng số trường; `b` chuẩn hoá độ dài từng trường riêng | chỉ Lucene làm được; `bm25s` không có fielded BM25 -> hai backend lệch nhau, không so sánh được; không tương ứng với một vector duy nhất bên dense |
| B. Một trường, lặp title (`title² + abstract`) | cùng một chuỗi cho Lucene, `bm25s` và encoder -> ba nhánh nhìn **cùng một văn bản**; đổi `title_weight` chỉ là một tham số | **xấp xỉ**: lặp title làm `|d|` dài ra nên hệ số `b` bị ảnh hưởng; không tách được điểm title/abstract |
| C. Hai lần chấm điểm rồi cộng | trọng số rõ ràng | phải chuẩn hoá hai thang điểm khác nhau (không có cơ sở); chậm gấp đôi; `bm25s` cũng không hỗ trợ sạch |

## Quyết định

Chọn **B**: `preprocess.text.weighted_text(title, abstract, title_weight)` trả về
`title` lặp `round(title_weight)` lần rồi tới `abstract`. Mặc định
`title_weight = 2.0` (khớp `configs/bm25.yaml: fields.title`).

Với backend Lucene, văn bản này vẫn được đánh chỉ mục **một trường** (`contents`),
không dùng `--fields` của Pyserini.

## Lý do

1. **Ba nhánh phải nhìn cùng một văn bản.** Nếu BM25 đọc `title` + `abstract`
   theo cách khác encoder thì mọi so sánh "lexical vs dense" đều nhiễu. Một
   chuỗi duy nhất loại bỏ hẳn nguồn nhiễu đó.
2. **`bm25s` không có fielded BM25** — chọn A là tự tách đôi codebase thành hai
   hành vi khác nhau, trong khi `bm25s` chính là đường chạy khi thiếu Java.
3. **Đổi trọng số không phải dựng lại Doc Store**, chỉ dựng lại chỉ mục (~10
   phút với Lucene) — đủ nhanh để tune trong một buổi.

## Hệ quả

- Đây là **xấp xỉ**, phải ghi đúng chữ "xấp xỉ" ở báo cáo và slide: lặp title
  làm tăng `|d|` nên `b` không còn thuần là chuẩn hoá độ dài của abstract.
- `title_weight` là siêu tham số **thứ ba** bên cạnh `k1`, `b`; khi tune phải
  ghi lại cả ba trong `meta.json` (đã có) để run file truy được nguồn gốc.
- Bài chỉ có title (không abstract) không bị lặp title — nếu lặp, `|d|` sẽ bị
  thổi lên vô ích và bài đó bị phạt oan.
- Nếu sau này chuyển sang fielded BM25 thật, phải viết ADR mới thay thế ADR này
  và **đánh giá lại toàn bộ** điểm ablation (số cũ không so sánh được).
