"""Ứng dụng Flask.

Dạng trang kết quả:
    - ô tìm kiếm ở đầu trang
    - 10 kết quả/trang, phân trang
    - mỗi kết quả: tiêu đề (link PMC/DOI) | nguồn | snippet tô sáng | hạng
    - tab kỹ thuật: điểm RRF, rank từng nhánh, độ trễ từng tầng

Snippet tính theo TRANG, không tính trước cả 45 bài. Cache theo
(query_hash, cord_uid).

Chạy hoàn toàn cục bộ, không cần mạng.

Chưa cài đặt.
"""
