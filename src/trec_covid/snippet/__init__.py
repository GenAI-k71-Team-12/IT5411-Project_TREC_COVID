"""Trích dẫn mức câu (GĐ4).

Chạy SAU khi có thứ hạng cuối, CHỈ cho trang đang hiển thị (10 bài).
Chi phí: ~50 cặp (query, câu) mỗi trang, ~0,3 s trên GPU.

Bất biến:
    1. Nguyên văn — không sửa một chữ nào
    2. Không cắt giữa câu — quá dài thì cắt ở ~40 từ + "..."
    3. Không trộn bài — câu của bài A chỉ dưới tiêu đề bài A
    4. Không sinh văn bản — không dùng LLM tóm tắt/diễn giải
"""
