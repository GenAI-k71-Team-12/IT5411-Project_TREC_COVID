"""Ba công thức dung hợp, chạy trên CÙNG một tập ứng viên.

    (1) RRF chuẩn       RRF(d) = sum 1 / (k + rank(d)),        k = 60
    (2) Weighted RRF    RRF(d) = sum w / (k + rank(d))
    (3) Convex          s(d)   = a*bm25(d) + (1-a)*dense(d),   đã min-max

Vai trò — đây là thiết kế thí nghiệm, không phải ba lựa chọn ngang nhau:

    (1) đường cơ sở: chỉ dùng thứ hạng, không tinh chỉnh gì ngoài k
    (2) kiểm tra một nhánh có đáng tin hơn nhánh kia không
    (3) kiểm tra biên độ điểm có mang thông tin mà thứ hạng làm mất không

LƯU Ý: min-max tính TRONG TỪNG TRUY VẤN, không phải trên toàn corpus.

Chưa cài đặt.
"""
