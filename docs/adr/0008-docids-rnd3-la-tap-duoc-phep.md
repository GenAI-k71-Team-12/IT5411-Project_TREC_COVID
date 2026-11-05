# 0008: `docids-rnd3.txt` là danh sách được phép, không phải danh sách bị loại

- **Ngày:** 2026-11-05
- **Người:** Người 3 (Fusion · Rerank · Đánh giá)
- **Trạng thái:** chấp nhận

## Bối cảnh

Tầng postfilter (GĐ4) lọc run file theo `docids-rnd3.txt`. File này có thể hiểu theo
hai cách ngược nhau:

- **A. Danh sách được phép:** toàn bộ tài liệu hợp lệ của vòng 3. Run chỉ được chứa id
  nằm trong file.
- **B. Danh sách bị loại:** các bài đã chấm ở vòng 1–2. Run phải bỏ id nằm trong file.

Hiểu sai thì bộ lọc hoặc không làm gì, hoặc xoá gần hết run. Ngoài ra
`docs/ARCHITECTURE.md` §4.1 đang viết lẫn cả hai cách: "tập tài liệu hợp lệ của vòng,
đã loại các bài đã chấm ở vòng 1–2". Vì vậy phải đếm, không đoán.

## Bằng chứng (đếm ngày 2026-11-05)

Chạy từ gốc repo, với dữ liệu đặt đúng chỗ như `data/README.md` mô tả:

```powershell
# 1. Số dòng và số id duy nhất trong docids-rnd3.txt
python -c "ids=[l.strip() for l in open('data/external/docids-rnd3.txt') if l.strip()]; print(len(ids), len(set(ids)))"

# 2. So với Doc Store (bản 19/05/2020 dựng từ metadata.csv)
python -c "import pandas as pd; D=set(l.strip() for l in open('data/external/docids-rnd3.txt') if l.strip()); S=set(pd.read_parquet('data/processed/doc_store.parquet',columns=['cord_uid']).cord_uid); print(len(D&S), len(D-S), len(S-D))"

# 3. Bài đã chấm ở vòng 1-2 (qrels.csv) có nằm trong docids không
python -c "import pandas as pd; D=set(l.strip() for l in open('data/external/docids-rnd3.txt') if l.strip()); q=pd.read_csv('data/external/qrels.csv',dtype=str); J=set(q['cord-id']); print(sorted(q.iteration.unique()), len(J), len(J&D), len(J-D))"
```

| Lệnh | Kết quả |
|---|---|
| 1 | 128.492 dòng, **128.162** id duy nhất (330 dòng lặp) |
| 2 | giao = **128.162**, docids − store = **0**, store − docids = **0** |
| 3 | iteration `0.5, 1.0, 1.5, 2.0`; 10.287 bài đã chấm, **9.822 nằm trong docids**, 465 không |

Đọc kết quả:

- Lệnh 2: docids trùng khít với toàn bộ Doc Store. Nếu đây là danh sách bị loại (B) thì
  phải loại sạch collection, điều đó vô lý.
- Lệnh 3: phần lớn bài đã chấm ở vòng 1–2 **vẫn nằm trong** docids, nên file này không
  loại bài đã chấm. 465 bài còn lại là bài có trong qrels nhưng đã bị gỡ khỏi CORD-19
  bản 19/05 (không có cả trong Doc Store).
- `qrels.csv` của cuộc thi chỉ có phán định **vòng 1–2** (iteration ≤ 2.0), 35 topic.

## Phương án đã cân nhắc

| Phương án | Ưu | Nhược |
|---|---|---|
| A. Được phép: giữ id trong file | khớp cả 3 lệnh đếm | lọc là no-op trên Doc Store hiện tại |
| B. Bị loại: bỏ id trong file | — | mâu thuẫn lệnh 2: xoá toàn bộ run |

## Quyết định

`docids-rnd3.txt` là **danh sách được phép** (A). Bộ lọc theo file này giữ id nằm trong
file. Với Doc Store hiện tại, bộ lọc không loại bài nào.

"Residual" đúng nghĩa của TREC-COVID là một việc **khác**: với mỗi topic, bỏ các bài đã
có phán định ở vòng trước cho **chính topic đó**. Nguồn của nó là `qrels.csv`, không
phải `docids-rnd3.txt`. Chế độ này và số đo hai chiều để ADR 0005 quyết định.

## Hệ quả

- Dòng "Phải lọc residual theo `docids-rnd3.txt` trước khi ghi run" của ADR 0002 được
  hiểu là lọc theo danh sách được phép. Việc này đúng thể lệ nhưng không đổi điểm.
- `docs/ARCHITECTURE.md` §4.1 sửa lại theo ADR này.
- 465 bài có phán định nhưng không có trong Doc Store thì không hệ thống nào của nhóm
  truy hồi được, nên đây là trần recall cố định. Báo cáo cần nhắc khi so điểm.
- Khi chấm cục bộ bằng `qrels.csv` (vòng 1–2), lọc residual theo qrels sẽ xoá chính các
  bài có nhãn. Điểm sụp không phải vì hệ thống kém. Xem ADR 0005.
- Nếu đổi sang bản CORD-19 khác thì phải đếm lại: kết luận "no-op" chỉ đúng khi Doc
  Store được dựng từ đúng bản 19/05/2020.
