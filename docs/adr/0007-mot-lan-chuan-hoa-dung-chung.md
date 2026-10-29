# 0007 — Một làn chuẩn hoá an toàn dùng chung cho BM25 và encoder

- **Ngày:** 2026-10-29
- **Người:** Người 1 (Dữ liệu & BM25)
- **Trạng thái:** chấp nhận

## Bối cảnh

CORD-19 là tập dữ liệu thu thập tự động từ hàng chục nhà xuất bản, nên văn bản
bẩn theo những cách rất cụ thể:

- `COVID‑19` với NBSP (U+00A0) hoặc 4 loại gạch nối Unicode khác nhau
- HTML entity còn sót, có chỗ bị escape hai lần (`&amp;amp;`)
- Placeholder `[Figure 1]`, `[Table 2]` nằm ngay trong abstract
- Rác bản quyền lặp ở hàng nghìn bài (`© 2020 Elsevier Ltd. All rights reserved.`)
- `µ` (micro sign U+00B5) và `μ` (Greek mu U+03BC) dùng lẫn
- Zero-width space chen giữa từ

Đồng thời pipeline có **hai nhánh** cùng đọc văn bản này: BM25 (Lucene, có
stemming + stopword) và encoder (PubMedBERT, cần câu tự nhiên).

## Phương án đã cân nhắc

| Phương án | Ưu | Nhược |
|---|---|---|
| A. Mỗi nhánh tự chuẩn hoá | mỗi bên tự do tối ưu | hai bên lệch nhau -> truy vấn đến BM25 và đến encoder khác nhau, phép so sánh lexical vs dense mất nghĩa; sửa bug phải sửa hai chỗ |
| B. Một làn "an toàn" dùng chung, việc nặng (stem, stopword) để tầng analyzer | văn bản hai nhánh giống hệt nhau; encoder không bao giờ nhận token đã stem | phải phân định rõ việc nào "an toàn" (không đổi nghĩa) và việc nào không |

## Quyết định

Chọn **B**, hiện thực trong `preprocess/text.py`:

| Hàm | Việc | Dùng cho |
|---|---|---|
| `normalize_document()` | NFC, NBSP -> dấu cách, gạch nối Unicode -> `-`, bỏ zero-width/control, giải HTML entity (lặp tới ổn định), bỏ placeholder hình/bảng, bỏ rác bản quyền, `µ` -> `μ`, gộp khoảng trắng | Doc Store, văn bản đưa vào Lucene **và** encoder |
| `normalize_query()` | y hệt `normalize_document` (cố ý) | cả hai nhánh truy vấn |
| `query_terms()` | thêm lowercase + bỏ stopword (+ stem tuỳ chọn) | `bm25s`, tô sáng keyword ở snippet |
| `escape_lucene()` | escape ký tự đặc biệt của QueryParser | **chỉ** khi bật raw query parser |

Không lowercase, không stem, không bỏ stopword trong làn an toàn.

## Lý do

1. **NBSP là bẫy im lặng nguy hiểm nhất.** `Character.isWhitespace(U+00A0)` của
   Java trả `false`, nên StandardTokenizer **không** cắt tại NBSP: Lucene sinh
   token `covid\u00a019` và không bao giờ khớp `covid`. Không có lỗi, chỉ mất
   recall — loại lỗi khó phát hiện nhất khi chấm điểm.
2. **Rác bản quyền làm sai IDF.** Một câu như "All rights reserved" xuất hiện ở
   hàng nghìn bài sẽ bị BM25 coi là thuật ngữ phổ biến, kéo IDF của các từ
   chung quanh xuống.
3. **Encoder không được nhận token đã stem.** PubMedBERT được huấn luyện trên
   câu tự nhiên; đưa `patient` (đã stem từ `patients`) vào là lệch phân phối.
   Vì vậy stemming/stopword **chỉ** nằm ở analyzer của Lucene hoặc ở
   `query_terms()`.
4. **Không escape mặc định.** `LuceneSearcher.search()` tự dựng BooleanQuery từ
   token đã phân tích; escape sẵn sẽ biến `\\(` thành token rác và làm hỏng cả
   chuỗi gửi cho encoder.

## Hệ quả

- `normalize_query()` **phải** cho ra đúng chuỗi `normalize_document()` cho ra —
  có test khoá bất biến này. Nếu sau này thêm bước riêng cho truy vấn, phải
  thêm ADR.
- Sửa `normalize_document()` là **thay đổi dữ liệu**: Doc Store, chỉ mục Lucene
  và FAISS đều phải dựng lại, và mọi điểm đã đo trước đó không so sánh được.
  Vì vậy hàm này chỉ được sửa khi thật cần, và phải chạy lại
  `tests/test_preprocess.py`.
- Bộ lọc rác bản quyền là **danh sách mẫu**, không phải bộ phân loại: nó bỏ
  sót và có thể bỏ oan. Chấp nhận, vì bỏ sót chỉ làm nhiễu nhẹ, còn bỏ oan một
  câu kết luận thì hiếm (mẫu đều neo vào `©`, `copyright`, `all rights reserved`).
- `µ` -> `μ` là quyết định có chủ ý: nếu để nguyên, `5 µg` và `5 μg` là hai
  token khác nhau dù người đọc hiểu như một.
