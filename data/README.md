# Dữ liệu

**Không commit gì trong thư mục này** (trừ `.gitkeep` và file README).

| Thư mục | Nội dung | Kích thước |
|---|---|---|
| `raw/` | CORD-19 bản 19/05/2020: `metadata.csv` + `pdf_json/` + `pmc_json/` | ~12.7 GB |
| `interim/` | Trung gian sau khi parse, trước khi chuẩn hoá | vài GB |
| `processed/` | Doc Store đã chuẩn hoá (`doc_store.parquet`) | vài trăm MB |
| `external/` | File của cuộc thi (nhỏ) | vài MB |

## Lấy dữ liệu

| File | Nguồn | Đặt vào |
|---|---|---|
| CORD-19 19/05/2020 | trang của cuộc thi / AI2 | `raw/cord19-2020-05-19/` |
| `topics-rnd3.csv` | tab Data | `external/` |
| `docids-rnd3.txt` | tab Data | `external/` |
| `qrels.csv` | tab Data | `external/` |

## Kiểm tra sau khi đặt

```powershell
Get-ChildItem data\raw\cord19-2020-05-19 | Select-Object Name, Length
Get-Content data\external\docids-rnd3.txt | Measure-Object -Line
Get-Content data\external\qrels.csv | Measure-Object -Line
```

`docids-rnd3.txt` phải có hàng chục nghìn dòng. Nếu chỉ vài nghìn thì đã lấy
nhầm file của vòng khác.

Kiểm tra qrels có đúng 35 topic có phán định:

```powershell
Get-Content data\external\qrels.csv | ForEach-Object { ($_ -split ' ')[0] } | Sort-Object -Unique
```
