"""Hệ thống tìm kiếm y sinh Hybrid Search trên TREC-COVID.

Luồng dữ liệu một chiều:

    io -> preprocess -> index -> retrieve -> fuse -> rerank -> postfilter
                                                                |
                                                  snippet / evaluate / web

Quy tắc: module sau được import module trước, không được import ngược.
"""

__version__ = "0.1.0"
