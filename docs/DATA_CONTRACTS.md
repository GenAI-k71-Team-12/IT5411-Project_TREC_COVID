# Quy ước giao tiếp dữ liệu giữa các thành viên (Data Contracts)

Tài liệu này quy định **hợp đồng dữ liệu (Data Contracts / Interface Specifications)** bắt buộc giữa 4 thành viên phụ trách các phần độc lập. Tất cả các thành viên phải tuân thủ chuẩn này (được code hoá tại [`src/trec_covid/schemas.py`](../src/trec_covid/schemas.py)) để tránh tình trạng vỡ hệ thống khi tích hợp pipeline.

---

## 1. Sơ đồ dòng chảy dữ liệu (Dataflow)

```
                            [Người 1: BM25]
                           search(query, 1000)
                                    │
                                    ├──► List[ScoredDocument] ──┐
                                    │                           │
User Query ──► [Preprocess] ────────┤                           ├─► [Người 3: Fusion & Rerank]
                                    │                           │     1. fuse(bm25, dense)
                                    ├──► List[ScoredDocument] ──┘     2. rerank(top-50)
                                    │                                           │
                             [Người 2: Dense]                                   ▼
                           search(query, 1000)                        List[ScoredDocument]
                                                                         (top-1000 final)
                                                                                │
                                                                                ▼
[Người 4: Web UI Flask]  ◄────── Page Items (top-10) ────────────── [Paging & Doc Store]
   (Hiển thị danh sách card)               │
              ▲                            ▼
              │                  [Người 4: Snippet]
              └─────────────── generate_snippets(page_docs)
```

---

## 2. Chi tiết hợp đồng từng phân hệ

### 2.0. Doc Store — Người 1 (điều kiện tiên quyết của cả hai nhánh)

- **Class tương ứng:** `DocumentRecord`
- **File lưu trữ:** `data/processed/doc_store.parquet`
- **Các trường bắt buộc:**

| Trường | Kiểu | Mô tả | Ví dụ |
|---|---|---|---|
| `cord_uid` | `str` | Mã định danh bài báo (Primary Key) | `"ug7v899j"` |
| `title` | `str` | Tiêu đề bài báo | `"Clinical features of patients infected with 2019 novel coronavirus"` |
| `abstract` | `str` | Tóm tắt nguyên văn chuỗi (không tách câu trước) | `"Coronaviruses are important human and animal pathogens..."` |
| `authors` | `str` | Tác giả (chuỗi phân cách bằng dấu `;`) | `"Huang, Chaolin; Wang, Yeming; ..."` |
| `journal` | `str` | Tạp chí xuất bản | `"Lancet"` |
| `publish_time`| `str` | Năm hoặc ngày xuất bản | `"2020-02-15"` |
| `doi` | `str` | Mã định danh số tài liệu | `"10.1016/S0140-6736(20)30183-5"` |
| `pmcid` | `str` | Mã định danh PubMed Central | `"PMC7159299"` |

> **Quy tắc:** Nếu bài viết không có abstract, trường `abstract` phải là chuỗi rỗng `""` (không dùng `NaN` hay `None`).

> ⏰ **Bàn giao hai đợt (quan trọng cho tiến độ):**
> - **Đợt 1 — trong 2 ngày đầu:** Doc Store **tối giản 3 trường** `cord_uid`, `title`, `abstract` **+ `docid_order.parquet`** (sinh ngay, vì FAISS cần `position -> cord_uid` để Người 2 dựng chỉ mục). Đây là **mốc chặn**: Người 2 cần nó để bắt đầu encode ~170k bài (2–3 giờ), không thể chờ đủ 8 trường.
> - **Đợt 2 — sau đó:** ghi đè **cùng đường dẫn** bằng bản đủ `authors`, `journal`, `publish_time`, `doi`, `pmcid`. Các trường này **không tham gia vào vector**, thứ tự dòng **không đổi**, nên **không phải encode lại**.
>
> **Bất biến bắt buộc:** thứ tự dòng của Doc Store phải tất định. FAISS trả **vị trí số nguyên**, nên `docid_order.parquet` (`position`, `cord_uid`) phải được sinh cùng lúc và `assert len(docids) == index.ntotal`. Không ai được `sort_values()` lại Doc Store sau khi đã encode.

---

### 2.1. Người 1 — BM25 (Lexical Retrieval)

- **Đầu vào:**
  - `query: str`: Câu truy vấn tìm kiếm của người dùng (đã qua tiền xử lý cơ bản).
  - `top_k: int = 1000`: Số lượng ứng viên cần trả về.
- **Đầu ra:** `List[ScoredDocument]`
- **Quy ước:**
  - Mỗi phần tử có `cord_uid: str`, `score: float` (điểm Lucene BM25 — **thang không chuẩn hoá**, phụ thuộc `k1`/`b` và độ dài văn bản; **không** so sánh trực tiếp với điểm của nhánh dense), `rank: int` (1-indexed, từ 1 đến 1000), `source: str = "bm25"`.
  - Danh sách **bắt buộc sắp xếp giảm dần theo `score`**.
- **Chữ ký hàm chuẩn:**
  ```python
  def search_bm25(query: str, top_k: int = 1000) -> list[ScoredDocument]:
      ...
  ```
  (Bản đã cài đặt nhận thêm hai tham số **keyword-only** `index_dir=` và `config=`
  để test/ablation trỏ vào chỉ mục khác — không phá vỡ chữ ký trên.)

- **Đọc topic của cuộc thi:** `trec_covid.io.topics` — Người 1 giữ, Người 3 dùng
  ```python
  from trec_covid.io.topics import load_topics, build_queries

  topics = load_topics("data/external/topics-rnd3.csv")
  pairs = build_queries(topics, field=cfg["competition"]["query_field"])
  # -> [(topic_id, văn bản truy vấn), ...] đã bỏ topic có văn bản rỗng
  ```
  `load_topics` **không** chuẩn hoá văn bản: chuẩn hoá nằm trong `search_bm25` /
  `search_dense` để cả hai nhánh đi qua đúng một làn xử lý (ADR
  [0007](adr/0007-mot-lan-chuan-hoa-dung-chung.md)).
  Trường `query` / `question` / `narrative` chọn bằng
  `competition.query_field` trong `configs/base.yaml` — **chưa chốt**, phải đo.

---

### 2.2. Người 2 — Dense Vector (Semantic Retrieval)

- **Đầu vào:**
  - `query: str`: Câu truy vấn tìm kiếm của người dùng.
  - `top_k: int = 1000`: Số lượng ứng viên cần trả về.
- **Đầu ra:** `List[ScoredDocument]`
- **Quy ước:**
  - Mỗi phần tử có `cord_uid: str`, `score: float` (Cosine similarity từ FAISS `IndexFlatIP` sau khi chuẩn hoá L2), `rank: int` (1 đến 1000), `source: str = "dense"`.
  - Danh sách **bắt buộc sắp xếp giảm dần theo `score`**.
- **Chữ ký hàm chuẩn:**
  ```python
  def search_dense(query: str, top_k: int = 1000) -> list[ScoredDocument]:
      ...
  ```

> ⚠️ **Bẫy chuẩn hoá cho Người 3:** về lý thuyết cosine nằm trong `[-1.0, 1.0]`, nhưng điểm MedCPT thực tế **co cụm rất hẹp, khoảng `[0.7, 0.95]`**. Nếu min-max theo `[-1, 1]`, chênh lệch thật `0,004` sẽ bị phóng thành `0,126` — Convex min-max vì thế phải chuẩn hoá **theo min/max của chính danh sách trong từng truy vấn**, không dùng hằng số.

---

### 2.3. Người 3 — Dung hợp (Fusion) & Tái xếp hạng (Reranker)

- **Đầu vào:**
  - `bm25_results: list[ScoredDocument]`: Top-1000 của Người 1.
  - `dense_results: list[ScoredDocument]`: Top-1000 của Người 2.
  - `query: str`: Câu truy vấn.
  - `rerank_top_k: int = 50`: Số lượng ứng viên lấy sau fusion để đưa vào Cross-Encoder.
- **Quy trình Người 3 thực hiện:**
  1. Hợp nhất hai danh sách bằng RRF / Weighted RRF / Convex -> Ra danh sách xếp hạng ~1000–1800 tài liệu duy nhất.
  2. Lấy `top-50` bài cao nhất sau dung hợp, nạp `(title, abstract)` từ Doc Store tạo thành `CandidateForRerank`.
  3. Đưa qua **MedCPT Cross-Encoder** chấm điểm `rerank_score`.
  4. Ghép lại danh sách kết quả cuối: Top-50 bài được sắp xếp theo `rerank_score`, các bài từ 51 trở đi giữ nguyên thứ tự dung hợp (hoặc normalize điểm nối tiếp để đảm bảo tính **đơn điệu giảm**).
  5. Lọc bỏ các bài ngoài `docids-rnd3.txt` (nếu chạy tạo run file).
- **Đầu ra:** `List[ScoredDocument]` (chứa 1000 bài có thứ hạng cuối cùng, `score` giảm dần, `rank` từ 1 đến 1000).
- **Chữ ký hàm chuẩn:**
  ```python
  def fuse_and_rerank(
      query: str,
      bm25_results: list[ScoredDocument],
      dense_results: list[ScoredDocument],
      rerank_top_k: int = 50
  ) -> list[ScoredDocument]:
      ...
  ```

---

### 2.4. Người 4 — Trích đoạn & Tô sáng (Snippet Generation)

- **Đầu vào:**
  - `query: str`: Truy vấn của người dùng.
  - `documents: list[DocumentRecord]`: Danh sách các bài báo hiển thị trên trang hiện tại (thường là 10 bài của trang đang xem).
- **Đầu ra:** `dict[str, SnippetResult]` (Key là `cord_uid`, Value là đối tượng `SnippetResult`).
- **Quy ước:**
  - `sentence`: Câu nguyên văn được chọn từ abstract có liên quan nhất tới query.
  - `highlighted_html`: Chuỗi câu nguyên văn với các từ khoá trùng khớp hoặc liên quan được bọc trong thẻ `<mark class="hl">...</mark>`.
  - Nếu bài viết **không có abstract** (`has_abstract=False`): `sentence = ""` và `highlighted_html = '<span class="text-muted">Không có tóm tắt</span>'`.
- **Chữ ký hàm chuẩn:**
  ```python
  def generate_snippets(
      query: str,
      documents: list[DocumentRecord]
  ) -> dict[str, SnippetResult]:
      ...
  ```

---

### 2.5. Người 4 — Giao diện Web (Flask UI)

- **Đầu vào:** Nhận kết quả từ Người 3 (danh sách xếp hạng) + Doc Store (Người 1) + Snippet (Người 4).
- **Mô hình kết quả trả ra hiển thị (View Model):** `SearchResultItem`
  ```python
  @dataclass
  class SearchResultItem:
      cord_uid: str
      rank: int
      score: float
      title: str
      authors: str
      journal: str
      publish_time: str
      doi: str
      abstract: str
      snippet_html: str
      snippet_text: str
      has_abstract: bool = True
  ```
- **Trách nhiệm:** Người 4 sở hữu toàn bộ giao diện web (HTML/CSS/JS, Flask backend, phân trang, hiển thị điểm số và chế độ so sánh BM25 vs Dense vs Hybrid). Các thành viên khác đóng góp qua PR, không sửa trực tiếp.
- **Ràng buộc offline:** toàn bộ CSS/JS/font phải nằm cục bộ trong `src/trec_covid/web/static/` — **cấm CDN**, vì buổi demo có thể diễn ra khi máy không có mạng.

---

## 3. Dữ liệu giả lập (Mock Data) để kiểm thử độc lập

Mỗi thành viên có thể copy đoạn mã dưới đây vào file test của mình để chạy thử code mà **không cần chờ các thành viên khác hoàn thành**:

```python
from trec_covid.schemas import ScoredDocument, DocumentRecord, SnippetResult

# Dữ liệu giả lập đầu ra BM25 / Dense (dành cho Người 3 test Fusion & Rerank)
MOCK_BM25_RESULTS = [
    ScoredDocument(cord_uid="doc_001", score=18.5, rank=1, source="bm25"),
    ScoredDocument(cord_uid="doc_002", score=16.2, rank=2, source="bm25"),
    ScoredDocument(cord_uid="doc_003", score=14.1, rank=3, source="bm25"),
]

MOCK_DENSE_RESULTS = [
    ScoredDocument(cord_uid="doc_002", score=0.88, rank=1, source="dense"),
    ScoredDocument(cord_uid="doc_001", score=0.82, rank=2, source="dense"),
    ScoredDocument(cord_uid="doc_004", score=0.79, rank=3, source="dense"),
]

# Dữ liệu giả lập Doc Store (dành cho Người 4 test Snippet & Nhóm test UI)
MOCK_DOCUMENTS = [
    DocumentRecord(
        cord_uid="doc_001",
        title="Clinical characteristics of coronavirus disease 2019 in China",
        abstract="Coronavirus disease 2019 (COVID-19) emerged in Wuhan. Symptoms include fever and cough.",
        authors="Guan, W.; Ni, Z.; Hu, Y.",
        journal="New England Journal of Medicine",
        publish_time="2020",
        doi="10.1056/NEJMoa2002032"
    ),
    DocumentRecord(
        cord_uid="doc_002",
        title="Remdesivir in adults with severe COVID-19: a randomised trial",
        abstract="Remdesivir was not associated with statistically significant clinical benefits. More trials are needed.",
        authors="Wang, Y.; Zhang, D.; Du, G.",
        journal="Lancet",
        publish_time="2020",
        doi="10.1016/S0140-6736(20)31022-9"
    )
]
```

---

## 4. Xử lý các trường hợp ngoại lệ (Edge Cases)

1. **Truy vấn không trả về kết quả nào (Empty retrieval):** Nếu BM25 hoặc Dense không tìm thấy ứng viên, hàm phải trả về danh sách rỗng `[]`, không được raise Exception.
2. **Abstract rỗng:** Không được làm vỡ pipeline. Snippet generator phải gán `has_abstract = False` và hiển thị thông báo thay thế.
3. **Ký tự đặc biệt trong Query:** Cần được escape phù hợp trước khi đưa vào Lucene hoặc tokenizer (tránh crash khi gặp dấu `+`, `-`, `&&`, `||`, `!`, `(`, `)`...).
4. **Viết tắt y sinh khi tách câu:** Xử lý danh sách từ như: `et al.`, `i.e.`, `e.g.`, `Fig.`, `No.`, `vs.`, `Dr.`, `0.5 mg.` không bị tách nhầm thành 2 câu.
