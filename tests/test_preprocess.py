"""Test tầng chuẩn hoá văn bản — lá chắn duy nhất chống bẫy dữ liệu CORD-19."""

from __future__ import annotations

import pytest

from trec_covid.preprocess.text import (
    escape_lucene,
    normalize_document,
    normalize_query,
    query_terms,
    tokenize,
    weighted_text,
)


class TestNormalizeDocument:
    def test_nbsp_khong_lam_tach_token(self):
        # Bẫy nguy hiểm nhất: không lỗi, chỉ mất recall. Java coi U+00A0 là
        # KHÔNG phải khoảng trắng (Character.isWhitespace = false) nên Lucene
        # sẽ sinh token "covid\u00a019" và không bao giờ khớp "covid".
        assert normalize_document("COVID\u00a019") == "COVID 19"
        assert tokenize("COVID\u00a019") == tokenize("COVID-19")

    @pytest.mark.parametrize(
        "raw",
        [
            "COVID\u201019",  # hyphen U+2010
            "COVID\u201119",  # non-breaking hyphen U+2011
            "COVID\u201319",  # en dash
            "COVID\u201419",  # em dash
            "COVID\u221219",  # minus sign
            "COVID\u002d19",  # ASCII hyphen
        ],
    )
    def test_moi_loai_gach_noi_ve_cung_mot_dang(self, raw):
        assert normalize_document(raw) == "COVID-19"

    def test_giai_html_entity(self):
        assert normalize_document("A &amp; B") == "A & B"
        assert normalize_document("p &lt; 0.05") == "p < 0.05"

    def test_entity_bi_escape_hai_lan(self):
        assert normalize_document("A &amp;amp; B") == "A & B"

    def test_bo_placeholder_hinh_bang(self):
        out = normalize_document("Patients improved. [Figure 1] See [Table 2] for details.")
        assert "[Figure 1]" not in out
        assert "[Table 2]" not in out
        assert "Patients improved." in out

    def test_bo_rac_ban_quyen(self):
        out = normalize_document("Findings here. \u00a9 2020 Elsevier Ltd. All rights reserved.")
        assert "Elsevier" not in out
        assert "rights reserved" not in out
        assert "Findings here." in out

    def test_gop_khoang_trang(self):
        assert normalize_document("a\n\n  b\t\tc") == "a b c"

    def test_zero_width_bi_bo(self):
        assert normalize_document("re\u200bmd\u200besivir") == "remdesivir"

    def test_micro_sign_ve_greek_mu(self):
        assert normalize_document("5 \u00b5g") == "5 \u03bcg"

    def test_nfc(self):
        decomposed = "e\u0301"
        assert normalize_document(decomposed) == "\u00e9"

    @pytest.mark.parametrize("value", [None, ""])
    def test_dau_vao_rong(self, value):
        assert normalize_document(value) == ""

    def test_khong_lowercase_khong_stem(self):
        # Encoder phải nhận câu tự nhiên, không phải token đã xử lý.
        out = normalize_document("Remdesivir Was Not Associated")
        assert out == "Remdesivir Was Not Associated"


class TestNormalizeQuery:
    def test_giong_het_normalize_document(self):
        q = "Does remdesivir reduce mortality in COVID\u00a019?"
        assert normalize_query(q) == normalize_document(q)

    def test_khong_escape_mac_dinh(self):
        # Escape ở đây sẽ làm chuỗi gửi cho encoder bị lệch.
        assert normalize_query("COVID-19 (severe)") == "COVID-19 (severe)"


class TestEscapeLucene:
    @pytest.mark.parametrize("ch", list('+-&|!(){}[]^"~*?:\\/'))
    def test_escape_tung_ky_tu_dac_biet(self, ch):
        assert escape_lucene(ch) == "\\" + ch

    def test_cau_truy_van_co_ngoac(self):
        out = escape_lucene("COVID-19 (severe)")
        assert out == "COVID\\-19 \\(severe\\)"

    def test_chu_thuong_khong_doi(self):
        assert escape_lucene("remdesivir") == "remdesivir"


class TestQueryTerms:
    def test_bo_stopword(self):
        assert query_terms("does remdesivir reduce the mortality") == [
            "remdesivir",
            "reduce",
            "mortality",
        ]

    def test_lowercase(self):
        assert query_terms("Remdesivir") == ["remdesivir"]

    def test_bo_token_mot_ky_tu(self):
        assert "a" not in query_terms("a b cd")

    def test_stem_tuy_chon(self):
        assert query_terms("patients", stem=lambda w: w.rstrip("s")) == ["patient"]

    def test_rong(self):
        assert query_terms("") == []
        assert query_terms(None) == []


class TestWeightedText:
    def test_lap_title_theo_trong_so(self):
        out = weighted_text("T", "A", title_weight=2)
        assert out == "T T A"

    def test_trong_so_1(self):
        assert weighted_text("T", "A", title_weight=1) == "T A"

    def test_chi_co_title(self):
        assert weighted_text("T", "", title_weight=2) == "T"

    def test_chi_co_abstract(self):
        assert weighted_text("", "A", title_weight=2) == "A"

    def test_rong_ca_hai(self):
        assert weighted_text("", "") == ""
