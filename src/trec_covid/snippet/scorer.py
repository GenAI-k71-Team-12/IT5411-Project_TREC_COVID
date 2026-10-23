"""Chấm điểm (query, câu) và chọn câu trích dẫn.

Hai chế độ (configs/snippet.yaml):
    lexical        — đếm từ chung sau stemming, < 1 ms. Dùng khi chỉ có CPU.
    cross_encoder  — hiểu ngữ nghĩa, chấm theo batch (max_length = 128,
                     KHÔNG dùng 512 vì câu ngắn).

Ràng buộc:
    - câu được chọn phải chứa ít nhất 1 từ khoá của query
    - nếu điểm cao nhất dưới ngưỡng -> hiển thị 2 câu đầu abstract
    - tô sáng từ khoá bằng so khớp token, không dùng mô hình

Chưa cài đặt.
"""
