# 0004 — Snippet là trích nguyên văn, không sinh văn bản

- **Trạng thái:** chấp nhận

## Bối cảnh

Có thể dùng LLM để tóm tắt abstract thành một câu trả lời. Nhưng đó là
**sinh văn bản**, không phải **trích dẫn**.

## Phương án đã cân nhắc

| Phương án | Ưu | Nhược |
|---|---|---|
| LLM tóm tắt | đọc tự nhiên hơn | 2-10 giây/câu; có thể bịa; không truy ngược được câu gốc |
| Chọn 1 câu có sẵn | < 1 giây; kiểm chứng được; không thể bịa | đôi khi câu không trọn ý |

## Quyết định

Snippet là **một câu nguyên văn** trong abstract. Hệ thống chỉ **chọn** và
**tô sáng**, không sinh, không diễn giải, không cắt giữa câu.

## Bất biến

1. Nguyên văn — không sửa một chữ nào.
2. Không cắt giữa câu. Quá dài thì cắt ở ~40 từ + `...`, nhưng không cắt
   mất từ khoá.
3. Không trộn bài — câu của bài A chỉ xuất hiện dưới tiêu đề bài A.
4. Trên UI ghi rõ "trích nguyên văn từ abstract".

## Hệ quả

- Trả lời được câu hỏi phản biện: "Sản phẩm có sinh nội dung không?" -> Không.
- Không cần GPU cho phần này nếu dùng chấm điểm lexical.
- Nếu sau này muốn thêm "tóm tắt AI", phải ghi rõ đó là sinh tự động và
  tách khỏi phần trích dẫn.
