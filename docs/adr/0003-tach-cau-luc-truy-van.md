# 0003 — Tách câu lúc truy vấn, không lưu sentences[]

- **Trạng thái:** chấp nhận

## Bối cảnh

Snippet cần một câu trong abstract. Có hai chỗ để tách câu: lúc dựng chỉ mục
(GĐ1) hoặc lúc truy vấn (GĐ4).

## Phương án đã cân nhắc

| Phương án | Ưu | Nhược |
|---|---|---|
| Tách sẵn ở GĐ1, lưu `sentences[]` | lúc truy vấn nhanh hơn không đáng kể | phải lưu cho 170k bài (nếu lưu song song với `abstract` thì nhân đôi văn bản, ~250 MB); sửa splitter là phải chạy lại GĐ1 |
| Tách lúc truy vấn | không lưu gì thêm; sửa splitter thấy kết quả ngay | không có |

## Quyết định

Tách **lúc truy vấn**, chỉ cho 10 bài hiển thị (~10 ms). Doc Store chỉ giữ
`abstract` nguyên chuỗi.

## Lý do

1. **Không cần.** Mỗi trang chỉ hiển thị 10 bài.
2. **Khó đảo ngược.** Abstract y sinh đầy `et al.`, `0.5 mg.`, `Fig. 2` —
   splitter chắc chắn sai vài chỗ. Nếu đã nướng vào Doc Store thì sửa phải
   chạy lại GĐ1 (~2-3 giờ). Tách lúc truy vấn thì thấy kết quả ngay.
3. **Tránh phụ thuộc ẩn.** Nếu `sentences[]` nằm trong Doc Store, Doc Store
   trở thành hàm của phiên bản splitter, trong khi Lucene/FAISS không cần
   dựng lại -> phụ thuộc nửa vời, dễ gây nhầm.
4. **Một nguồn sự thật.** Logic tách câu chỉ tồn tại ở một chỗ.

## Hệ quả

- Slide 4 (kiến trúc) và slide 11 (sản phẩm) phải mô tả thống nhất: GĐ1
  không tách câu, bước Top-10 mới tách.
- Nếu sau này muốn truy xuất ở mức câu (index câu) thì phải xem lại ADR này.
