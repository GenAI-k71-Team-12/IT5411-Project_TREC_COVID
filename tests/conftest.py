"""Fixture dùng chung cho pytest.

Bộ dữ liệu nhỏ 6 bài, cố ý chứa đủ các bẫy của CORD-19 để test tầng dữ liệu
mà không cần nạp corpus thật (12,7 GB).
"""

from __future__ import annotations

import pandas as pd
import pytest

# Cố ý nhét bẫy: NBSP, gạch nối U+2011, HTML entity, placeholder hình,
# rác bản quyền, abstract rỗng, và một cord_uid trùng.
RAW_ROWS: list[dict[str, str]] = [
    {
        "cord_uid": "ug7v899j",
        "title": "Clinical features of patients infected with 2019 novel coronavirus",
        "abstract": (
            "BACKGROUND: In December 2019, pneumonia of unknown cause was detected in Wuhan. "
            "We report the clinical features of 41 patients with laboratory-confirmed "
            "2019\u2011nCoV infection. [Figure 1] All patients had pneumonia. "
            "\u00a9 2020 Elsevier Ltd. All rights reserved."
        ),
        "authors": "Huang, Chaolin; Wang, Yeming",
        "journal": "Lancet",
        "publish_time": "2020-02-15",
        "doi": "10.1016/S0140-6736(20)30183-5",
        "pmcid": "PMC7159299",
    },
    {
        "cord_uid": "02tnwd4m",
        "title": "Remdesivir and chloroquine effectively inhibit the recently emerged "
        "novel coronavirus (2019-nCoV) in vitro",
        "abstract": (
            "In late December 2019, a novel coronavirus was identified. "
            "Remdesivir (GS\u20115734) and chloroquine were tested in vitro. "
            "Both drugs inhibited the virus at low micromolar concentrations &amp; "
            "showed high selectivity index."
        ),
        "authors": "Wang, Manli",
        "journal": "Cell Research",
        "publish_time": "2020-02-04",
        "doi": "10.1038/s41422-020-0282-0",
        "pmcid": "PMC7054408",
    },
    {
        "cord_uid": "hjuf9m9s",
        "title": "Nosocomial transmission of severe acute respiratory syndrome coronavirus 2",
        "abstract": (
            "We describe nosocomial transmission of SARS-CoV-2 in a hospital ward in "
            "Wuhan. Health care workers were infected despite personal protective equipment."
        ),
        "authors": "Wang, Dawei",
        "journal": "JAMA",
        "publish_time": "2020-03-04",
        "doi": "10.1001/jama.2020.1585",
        "pmcid": "",
    },
    {
        "cord_uid": "a1b2c3d4",
        "title": "A title-only preprint without any abstract",
        "abstract": "",
        "authors": "",
        "journal": "bioRxiv",
        "publish_time": "2020-04-01",
        "doi": "",
        "pmcid": "",
    },
    {
        "cord_uid": "zzz99999",
        "title": "Dexamethasone in hospitalized patients with Covid-19",
        "abstract": (
            "BACKGROUND: Coronavirus disease 2019 is associated with substantial mortality. "
            "In this trial, dexamethasone resulted in lower mortality among patients "
            "receiving either invasive mechanical ventilation or oxygen alone."
        ),
        "authors": "Horby, Peter",
        "journal": "NEJM",
        "publish_time": "2020-07-17",
        "doi": "10.1056/NEJMoa2021436",
        "pmcid": "PMC7383595",
    },
    # Trùng cord_uid với bản đầu, nhưng abstract đầy đủ hơn -> phải giữ bản này
    {
        "cord_uid": "ug7v899j",
        "title": "Clinical features of patients infected with 2019 novel coronavirus",
        "abstract": (
            "BACKGROUND: In December 2019, pneumonia of unknown cause was detected in Wuhan. "
            "We report the clinical features of 41 patients with laboratory-confirmed "
            "2019-nCoV infection, including fever, cough, and lymphopenia."
        ),
        "authors": "Huang, Chaolin",
        "journal": "Lancet",
        "publish_time": "2020-02-15",
        "doi": "10.1016/S0140-6736(20)30183-5",
        "pmcid": "PMC7159299",
    },
]


@pytest.fixture
def raw_metadata_df() -> pd.DataFrame:
    """DataFrame thô giống hệt cái `load_cord19` trả về (mọi cột là str)."""
    return pd.DataFrame(RAW_ROWS).astype(str)


@pytest.fixture
def mini_metadata_csv(tmp_path, raw_metadata_df) -> object:
    """File `metadata.csv` thật trên đĩa, để test cả đường đọc file."""
    path = tmp_path / "metadata.csv"
    raw_metadata_df.to_csv(path, index=False, encoding="utf-8")
    return path


@pytest.fixture
def mini_doc_store(raw_metadata_df):
    """Doc Store đã sạch, đủ 8 trường."""
    from trec_covid.io.doc_store import build_doc_store

    return build_doc_store(raw_metadata_df)


@pytest.fixture
def mini_doc_store_path(tmp_path, mini_doc_store):
    """Doc Store đã ghi parquet."""
    from trec_covid.io.doc_store import save_doc_store

    return save_doc_store(mini_doc_store, tmp_path / "doc_store.parquet")


@pytest.fixture
def mini_index_dir(tmp_path, mini_doc_store):
    """Chỉ mục BM25 dựng bằng bm25s — không cần Java, chạy được trong CI."""
    from trec_covid.index.bm25_index import build_index

    out = tmp_path / "indexes" / "lucene"
    build_index(mini_doc_store, out, k1=0.9, b=0.4, backend="bm25s")
    return out
