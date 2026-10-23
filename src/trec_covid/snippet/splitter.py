"""Tách abstract thành câu — chạy lúc truy vấn, chỉ cho 10 bài.

Không cần lưu sentences[] cho 170k bài: tách 10 abstract mất ~10 ms.

Quy tắc cắt câu:
    1. Chỉ cắt khi ký tự đầu sau khoảng trắng là CHỮ HOA (hoặc "(")
    2. Không cắt nếu từ ngay trước dấu chấm là viết tắt y sinh
       (al, e.g, i.e, vs, Fig, No, Dr, approx, ref)
    3. Không cắt nếu ký tự ngay trước dấu chấm là CHỮ SỐ (0.5, 2.3)

Chưa cài đặt.
"""
