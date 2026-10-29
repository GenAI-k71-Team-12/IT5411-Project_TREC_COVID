"""Chuẩn hoá văn bản dùng chung cho cả hai nhánh truy xuất.

Nguyên tắc: phần **an toàn** (unicode NFC, NBSP, gạch nối Unicode, HTML entity,
gộp khoảng trắng, bỏ placeholder `[Figure 1]`) phải là MỘT hàm duy nhất dùng
chung cho cả hai nhánh — nếu lệch nhau thì query đến BM25 và query đến encoder
sẽ khác nhau, làm hỏng phép so sánh.

Stemming và bỏ stopword **chỉ** áp cho nhánh lexical, và chỉ ở tầng analyzer
(`EnglishAnalyzer` của Lucene tự làm). Encoder (PubMedBERT) **không bao giờ**
được nhận văn bản đã stem hay đã bỏ stopword.

Ba làn xử lý:

    normalize_document(t)      # làn an toàn — dùng cho Lucene VÀ encoder
    normalize_query(q)         # làn an toàn — dùng cho cả hai nhánh truy vấn
    query_terms(q)             # thêm: lowercase + bỏ stopword (+ stem nếu muốn)

`escape_lucene()` là hàm **riêng**, chỉ bật khi dùng raw query parser của
Lucene. Với `LuceneSearcher.search()` mặc định, Pyserini tự dựng BooleanQuery
từ token đã phân tích nên không cần escape.
"""

from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Callable, Iterable

# ---------------------------------------------------------------------------
# Bảng ký tự cần chuẩn hoá
# ---------------------------------------------------------------------------

#: Mọi loại khoảng trắng "lạ" — nguy hiểm nhất là U+00A0 (NBSP) vì nó làm
#: `COVID‑19` thành hai token mà không có lỗi nào báo ra.
_SPACE_CHARS = (
    "\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007"
    "\u2008\u2009\u200a\u202f\u205f\u3000"
)

#: CORD-19 trộn nhiều loại gạch nối: U+002D (ASCII), U+2010, U+2011 (non-breaking),
#: U+2012, U+2013 (en dash), U+2014 (em dash), U+2212 (minus), U+FE63, U+FF0D.
_HYPHEN_CHARS = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe63\uff0d"

_SPACE_RE = re.compile("[" + re.escape(_SPACE_CHARS) + "]")
_HYPHEN_RE = re.compile("[" + re.escape(_HYPHEN_CHARS) + "]")
_ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS_RE = re.compile(r"\s+")

#: Placeholder hình/bảng nằm ngay trong abstract của CORD-19.
_PLACEHOLDER_RE = re.compile(
    r"\[\s*(?:figure|fig\.?|table|scheme|supplementary\s+figure|video)\s*\d+[^\]]*\]",
    re.IGNORECASE,
)

#: Rác bản quyền — xuất hiện ở hàng nghìn bài nên nếu giữ lại thì IDF bị sai.
_BOILERPLATE_PATTERNS = (
    re.compile(r"©\s*\d{4}[^.]*\.?", re.IGNORECASE),
    re.compile(r"copyright\s*©?\s*\d{4}[^.]*\.?", re.IGNORECASE),
    re.compile(r"all rights reserved\.?", re.IGNORECASE),
    re.compile(r"published by [^.]*\.", re.IGNORECASE),
    re.compile(r"this article is protected by copyright[^.]*\.", re.IGNORECASE),
    re.compile(r"elsevier ltd\.?\s*all rights reserved\.?", re.IGNORECASE),
)

#: Ký tự đặc biệt của Lucene QueryParser — phải escape khi dùng raw parser.
LUCENE_SPECIAL_CHARS = frozenset('+-&|!(){}[]^"~*?:\\/')

#: Stopword tiếng Anh tối giản, đủ dùng cho `query_terms()` (highlight + bm25s).
#: Lucene dùng danh sách riêng của `EnglishAnalyzer`, không dùng bảng này.
STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after again against all am an and any are aren't as at be because been
    before being below between both but by can cannot could couldn't did didn't do does
    doesn't doing don't down during each few for from further had hadn't has hasn't have
    haven't having he her here hers herself him himself his how i if in into is isn't it
    its itself just let's me more most mustn't my myself no nor not of off on once only or
    other ought our ours ourselves out over own same shan't she should shouldn't so some
    such than that the their theirs them themselves then there these they this those
    through to too under until up very was wasn't we were weren't what when where which
    while who whom why with won't would wouldn't you your yours yourself yourselves
    """.split()
)

_TOKEN_RE = re.compile(r"\b\w\w+\b", re.UNICODE)


# ---------------------------------------------------------------------------
# Làn an toàn
# ---------------------------------------------------------------------------


def _strip_boilerplate(text: str) -> str:
    for pattern in _BOILERPLATE_PATTERNS:
        text = pattern.sub(" ", text)
    return text


def normalize_document(text: str | None) -> str:
    """Chuẩn hoá an toàn — dùng được cho cả Lucene và encoder.

    Làm đúng những việc không làm thay đổi ngữ nghĩa câu:

    1. Unicode NFC (`é` tổ hợp vs `é` dựng sẵn là hai token khác nhau)
    2. NBSP và các khoảng trắng lạ -> dấu cách thường
    3. Gạch nối Unicode -> `-` ASCII
    4. Zero-width và ký tự điều khiển -> bỏ
    5. Giải HTML entity (`&amp;` -> `&`), lặp tới khi ổn định
    6. Bỏ placeholder `[Figure 1]`, `[Table 2]`
    7. Bỏ rác bản quyền
    8. `µ` (micro sign U+00B5) -> `μ` (Greek mu U+03BC)
    9. Gộp mọi khoảng trắng thành một dấu cách, strip hai đầu

    Không lowercase, không stem, không bỏ stopword — đó là việc của analyzer
    hoặc của `query_terms()`.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _CONTROL_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text)
    text = _HYPHEN_RE.sub("-", text)
    text = text.replace("\u00b5", "\u03bc")

    # CORD-19 có trường hợp entity bị escape hai lần: "&amp;amp;"
    for _ in range(3):
        unescaped = html.unescape(text)
        if unescaped == text:
            break
        text = unescaped

    text = _PLACEHOLDER_RE.sub(" ", text)
    text = _strip_boilerplate(text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def normalize_query(query: str | None) -> str:
    """Chuẩn hoá truy vấn — **phải cho ra đúng chuỗi mà `normalize_document` cho ra**.

    Cố ý không escape ký tự Lucene: nếu escape ở đây thì chuỗi đưa vào encoder
    cũng dính dấu `\\`, và hai nhánh sẽ không còn so sánh được với nhau.
    Dùng `escape_lucene()` riêng khi thật sự chạy raw query parser.
    """
    return normalize_document(query)


def escape_lucene(text: str) -> str:
    """Escape ký tự đặc biệt của Lucene QueryParser.

    Chỉ cần khi bật raw query parser. Ví dụ topic TREC-COVID có
    `"COVID-19 (severe)"` — dấu ngoặc làm parser hiểu là nhóm Boolean và crash.
    """
    if not text:
        return ""
    return "".join("\\" + ch if ch in LUCENE_SPECIAL_CHARS else ch for ch in text)


# ---------------------------------------------------------------------------
# Làn lexical
# ---------------------------------------------------------------------------


def tokenize(text: str | None) -> list[str]:
    """Tách token thô: lowercase, `\\b\\w\\w+\\b` (bỏ token 1 ký tự)."""
    return _TOKEN_RE.findall(normalize_document(text).lower())


def query_terms(
    query: str | None,
    *,
    stem: Callable[[str], str] | None = None,
    stopwords: Iterable[str] | None = None,
    min_length: int = 2,
) -> list[str]:
    """Token của truy vấn cho nhánh lexical: lowercase + bỏ stopword (+ stem).

    Dùng cho:
      - `bm25s` (backend dự phòng, tự chấm điểm bằng token)
      - tô sáng keyword ở tầng snippet (Người 4)

    **Không** dùng cho Lucene — `EnglishAnalyzer` đã tự stem và bỏ stopword;
    đưa token đã stem vào Lucene là stem hai lần.

    Args:
        query: chuỗi truy vấn thô.
        stem: hàm stemming tuỳ chọn (vd. `PyStemmer.Stemmer("english").stemWord`).
        stopwords: danh sách stopword, mặc định dùng `STOPWORDS`.
        min_length: độ dài tối thiểu của token.
    """
    stops = set(STOPWORDS if stopwords is None else stopwords)
    terms = [t for t in tokenize(query) if len(t) >= min_length and t not in stops]
    if stem is not None:
        terms = [stem(t) for t in terms]
    return terms


def weighted_text(title: str | None, abstract: str | None, title_weight: float = 2.0) -> str:
    """Ghép `title` + `abstract` cho chỉ mục, có lặp title theo trọng số.

    CORD-19 có nhiều abstract dài, nên title thường bị "chìm". Lặp title
    `round(title_weight)` lần là cách xấp xỉ trọng số trường (fielded BM25)
    mà cả Lucene và `bm25s` đều hiểu giống nhau.

    Lưu ý: đây là **xấp xỉ**, không tương đương tuyệt đối với fielded BM25
    (việc lặp làm tăng `|d|` nên ảnh hưởng hệ số `b`).
    """
    t = normalize_document(title)
    a = normalize_document(abstract)
    if not a:
        return t
    if not t:
        return a
    repeat = max(1, int(round(float(title_weight))))
    return " ".join([t] * repeat + [a])
