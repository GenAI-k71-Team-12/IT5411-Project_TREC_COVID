"""Chỉ mục Lucene BM25 qua pyserini (fielded).

Cấu hình: configs/bm25.yaml
    - field title trọng số 2, abstract trọng số 1
    - EnglishAnalyzer (stemming Porter + stopword)
    - k1 = 0.9, b = 0.4 (hiệu chỉnh trên dev)

Dự phòng khi không cài được Java: bm25s (thuần Python, chậm hơn).

Chưa cài đặt.
"""
