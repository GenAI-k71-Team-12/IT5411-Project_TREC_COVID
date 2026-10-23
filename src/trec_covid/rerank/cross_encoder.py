"""MedCPT Cross-Encoder — checkpoint thứ ba, khác hai checkpoint bi-encoder.

Input: cặp (query, title + abstract), cắt 512 token.
Output: MỘT điểm 0-1 cho mỗi cặp (không phải vector) -> không cache được.

Chấm top-50, không phải top-1000 (170k cặp sẽ mất hàng giờ).

BẮT BUỘC: ghi điểm đơn điệu giảm theo thứ hạng cuối. Nếu lẫn thang điểm
với BM25 trong cùng cột score, trec_eval sẽ đảo thứ hạng.

Chưa cài đặt.
"""
