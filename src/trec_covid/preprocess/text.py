"""Chuẩn hoá văn bản dùng chung cho cả hai nhánh truy xuất.

Nguyên tắc: phần **an toàn** (unicode NFC, NBSP, gạch nối Unicode, HTML entity,
gộp khoảng trắng, bỏ placeholder `[Figure 1]`) phải là MỘT hàm duy nhất dùng
chung cho cả hai nhánh — nếu lệch nhau thì query đến BM25 và query đến encoder
sẽ khác nhau, làm hỏng phép so sánh.

Stemming và bỏ stopword **chỉ** áp cho nhánh lexical, và chỉ ở tầng analyzer
(`EnglishAnalyzer` của Lucene tự làm). Encoder (PubMedBERT) **không bao giờ**
được nhận văn bản đã stem hay đã bỏ stopword.

Interface dự kiến:

    def normalize_document(t: str) -> str: ...  # làn an toàn, dùng cho cả 2 nhánh
    def normalize_query(q: str) -> str: ...     # làn an toàn + escape ký tự Lucene
    def query_terms(q: str) -> list[str]: ...   # đã stemming, đã bỏ stopword

Chưa cài đặt.
"""
