# Quy ước đóng góp

Nhóm 4 người, làm song song trên 4 mảng phương pháp khác nhau. Tài liệu này là luật chung để tránh giẫm chân nhau.

---

## 1. Nhánh (branch)

Nhánh `main` **luôn chạy được**. Không commit trực tiếp lên `main`.

```
main                                   # bản ổn định, chỉ merge từ dev
 └─ dev                                # nhánh tích hợp của nhóm
     └─ feat/n<1-4>-<phạm-vi>          # tính năng mới
         fix/n<1-4>-<phạm-vi>          # sửa lỗi
         exp/n<1-4>-<thí-nghiệm>       # thí nghiệm, không merge vào dev
         docs/n<1-4>-<phạm-vi>         # tài liệu
```

Mã người: `n1` · `n2` · `n3` · `n4` — tương ứng Người 1–4 trong [`docs/TEAM.md`](docs/TEAM.md).

| Mã | Phụ trách |
|---|---|
| `n1` | Dữ liệu, Doc Store, BM25 |
| `n2` | Dense vector, FAISS, MedCPT |
| `n3` | Fusion, Rerank, Đánh giá |
| `n4` | Snippet, Web UI |

Ví dụ:

```
feat/n1-doc-store
feat/n1-bm25-retriever
feat/n2-faiss-index
feat/n3-rrf-fusion
fix/n2-faiss-l2norm
exp/n3-rerank-top100
docs/n4-adr-snippet
```

Nhánh sống tối đa **1 tuần**. Lâu hơn thì tách nhỏ lại.

---

## 2. Commit

Theo [Conventional Commits](https://www.conventionalcommits.org/), tiếng Việt hoặc tiếng Anh đều được nhưng **phải nhất quán trong một nhánh**.

```
<type>(<scope>): <mô tả ngắn>

[thân — tuỳ chọn]
```

`type`:

| type | Dùng khi |
|---|---|
| `feat` | thêm chức năng |
| `fix` | sửa lỗi |
| `refactor` | đổi cấu trúc, không đổi hành vi |
| `perf` | tối ưu tốc độ |
| `test` | thêm/sửa test |
| `docs` | tài liệu |
| `chore` | cấu hình, thư viện, dọn dẹp |
| `exp` | thí nghiệm (nhánh `exp/*`) |

`scope` lấy theo tên package trong `src/trec_covid/`: `index`, `retrieve`, `fuse`, `rerank`, `postfilter`, `snippet`, `evaluate`, `web`, `io`, `config`.

Ví dụ:

```
feat(index): dựng chỉ mục Lucene với field title^2 + abstract
fix(fuse): sửa RRF dùng rank 1-based thay vì 0-based
perf(rerank): gom batch 32 và cắt max_length 512
docs(adr): ghi quyết định tách câu lúc truy vấn
```

---

## 3. Pull request

1. Mở PR vào `dev` (không phải `main`), tiêu đề giống commit.
2. Điền template trong `.github/PULL_REQUEST_TEMPLATE.md`.
3. **Cần ít nhất 1 người khác approve.** Không tự merge PR của mình.
4. PR quá 400 dòng diff → tách nhỏ.
5. Merge bằng **squash and merge**, xoá nhánh sau khi merge.

### Checklist trước khi mở PR

- [ ] `pytest` xanh
- [ ] `ruff check src tests` không lỗi
- [ ] Không commit dữ liệu, chỉ mục, model, run file, `*.log`
- [ ] Cập nhật `configs/*.yaml` nếu có tham số mới (không hard-code)
- [ ] Cập nhật `docs/ARCHITECTURE.md` nếu đổi luồng xử lý
- [ ] Nếu là quyết định kiến trúc → thêm ADR trong `docs/adr/`

---

## 4. Phong cách code

- Python **3.10+**, type hint ở chữ ký hàm công khai.
- Định dạng bằng `ruff format`, dòng tối đa **100** ký tự.
- Tên biến/hàm bằng **tiếng Anh**; docstring và comment bằng **tiếng Việt**.
- Không hard-code đường dẫn — đọc từ `configs/*.yaml` hoặc biến môi trường (`.env`).
- Mỗi module chỉ làm một việc; hàm dài quá 50 dòng thì tách.

### Không được làm

| ❌ | Vì sao |
|---|---|
| `import` đường dẫn tuyệt đối của máy mình | máy người khác không chạy được |
| Commit `data/`, `indexes/`, `runs/`, `models/` | hàng GB, vượt giới hạn git |
| Sửa `configs/base.yaml` để thử nghiệm cá nhân | ảnh hưởng cả nhóm — tạo file riêng |
| Thay đổi tham số rồi báo cáo kết quả mà không ghi lại | không tái lập được |
| Dùng `rank_bm25` | quét O(N) mỗi truy vấn |

---

## 5. Dữ liệu và kết quả

- **Không bao giờ commit dữ liệu.** `data/`, `indexes/`, `models/` đã nằm trong `.gitignore`.
- Chia sẻ chỉ mục đã dựng qua ổ cứng ngoài / drive nhóm, kèm **mã băm** để kiểm tra toàn vẹn:

```powershell
Get-FileHash indexes\faiss\passage.index -Algorithm SHA256
```

- Kết quả `trec_eval` thì **được** commit (nhỏ, dạng text) vào `results/`, đặt tên theo cấu hình:

```
results/2026-11-02_rrf-k60_top50.txt
results/2026-11-02_ablation.csv
```

Mỗi kết quả phải kèm dòng đầu ghi rõ **commit hash + config** đã dùng, nếu không thì không so sánh được.

---

## 6. Họp nhóm

- **Thứ 2 hàng tuần, 30 phút**: mỗi người nêu (a) đã xong gì, (b) đang vướng gì, (c) tuần này làm gì.
- Vướng quá 2 giờ → nhắn nhóm, không tự loay hoay.
- Quyết định gì ảnh hưởng người khác → ghi vào `docs/adr/`, không chỉ nói miệng.
