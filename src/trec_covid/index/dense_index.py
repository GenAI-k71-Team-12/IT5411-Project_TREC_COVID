"""Encode corpus bằng MedCPT Article Encoder và nạp vào FAISS.

Các bước:
    1. Nạp Doc Store
    2. Encode (title + abstract) theo batch, max_length = 512
    3. Chuẩn hoá L2 vector        <-- BẮT BUỘC, nếu không IP != cos
    4. Nạp vào IndexFlatIP (không có bước train)
    5. Ghi xuống đĩa

Kích thước: 170k x 768 x 4 byte ~ 500 MB.
Thời gian encode: ~2-3 giờ (GPU).

Chưa cài đặt.
"""
