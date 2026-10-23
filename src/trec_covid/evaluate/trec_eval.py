"""Wrapper gọi binary trec_eval.

Các bẫy đã biết:
    - mặc định trec_eval KHÔNG tính nDCG -> phải -m ndcg_cut.10
    - không dùng -c khi chấm 40 topic mà qrels chỉ có 35:
      -c kéo các topic không có phán định vào trung bình với điểm 0
    - qrels chỉ có topic 1-35 -> điểm trung bình trên 35 topic

Chưa cài đặt.
"""
