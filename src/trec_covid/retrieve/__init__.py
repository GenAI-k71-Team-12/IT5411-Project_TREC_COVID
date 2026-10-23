"""Hai nhánh truy xuất (GĐ2). Mỗi nhánh trả về list[(docid, score)], top-1000.

Interface chung:

    def search(query: str, top_k: int = 1000) -> list[tuple[str, float]]: ...

Hai nhánh phải ĐỘC LẬP: không để nhánh này lọc ứng viên cho nhánh kia,
nếu không mất tính đa dạng mà fusion cần.
"""
