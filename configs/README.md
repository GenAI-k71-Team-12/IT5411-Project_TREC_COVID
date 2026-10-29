# Cấu hình

Mọi tham số nằm ở đây, **không hard-code trong code**. Lý do: cả nhóm chạy
ablation hàng chục lần, tham số phải truy vết được.

## Thứ tự merge

```
configs/base.yaml  ->  configs/<tầng>.yaml  ->  tham số truyền lúc chạy
```

Tham số sau ghi đè tham số trước.

## File

| File | Tầng | Nội dung |
|---|---|---|
| `base.yaml` | — | đường dẫn, file của cuộc thi, trường truy vấn (`query_field`), device |
| `bm25.yaml` | GĐ2 | `k1`, `b`, `title_weight` (một trường, title lặp), top-k |
| `dense.yaml` | GĐ2 | MedCPT, chiều vector, `IndexFlatIP` |
| `fusion.yaml` | GĐ3 | RRF / Weighted RRF / Convex |
| `rerank.yaml` | GĐ3 | Cross-Encoder, top-50 |
| `snippet.yaml` | GĐ4 | tách câu, chấm câu, cache |

## Quy ước

- **Không sửa `base.yaml` để thử nghiệm cá nhân** — tạo file riêng, ví dụ
  `configs/bm25_k1-1.2.yaml`, và ghi kết quả vào `results/`.
- Mỗi kết quả trong `results/` phải ghi rõ nó dùng config nào.
- Tham số nào đã hiệu chỉnh trên dev thì ghi chú lý do ngay cạnh giá trị.
