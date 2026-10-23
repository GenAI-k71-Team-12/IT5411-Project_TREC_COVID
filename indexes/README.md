# Chỉ mục

**Không commit.** Hàng trăm MB đến vài GB.

| Thư mục | Nội dung | Kích thước |
|---|---|---|
| `lucene/` | Chỉ mục BM25 (`title^2 + abstract`) | ~1-2 GB |
| `faiss/` | `IndexFlatIP` 170k x 768, float32 | ~500 MB |

## Đặc điểm

- **Tất định**: dùng cùng Doc Store + cùng config thì ra cùng chỉ mục.
  Nhờ vậy mọi người có thể tự dựng lại thay vì chép file.
- `IndexFlatIP` **không có bước train** (khác `IndexIVFFlat`), nên thời gian
  dựng = thời gian encode, không có ngẫu nhiên.
- Phải **chuẩn hoá L2 vector bài báo** trước khi nạp, nếu không `IP` không
  bằng `cosine`.

## Chia sẻ trong nhóm

Chép qua ổ cứng ngoài, kèm mã băm để kiểm tra:

```powershell
Get-FileHash indexes\faiss\articles.faiss -Algorithm SHA256
```

Ghi mã băm vào `docs/` để người khác đối chiếu. Mã khác nhau = một trong hai
bên đã dựng lại chỉ mục bằng cấu hình khác.
