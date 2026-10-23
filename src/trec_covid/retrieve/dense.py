"""Nhánh vector — encode query bằng MedCPT Query-Encoder rồi tìm trong FAISS.

LƯU Ý: dùng Query-Encoder (không dùng Article-Encoder) cho câu hỏi.
Phải là cặp checkpoint tương ứng với lúc encode corpus.

Mạnh: diễn đạt lại, câu hỏi tự nhiên.
Yếu: token hiếm bị chẻ sub-word rồi nhoè.

Chưa cài đặt.
"""
