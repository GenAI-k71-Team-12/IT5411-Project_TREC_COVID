"""Test đọc/merge cấu hình — mọi script đều đi qua đây trước khi làm gì."""

from __future__ import annotations

from pathlib import Path

import pytest

from trec_covid.config import _deep_merge, load_config, resolve_any_path, resolve_path

MINI_BASE = """
paths:
  doc_store: data/processed/doc_store.parquet
  lucene_index: indexes/lucene
competition:
  topics: data/external/topics-rnd3.csv
fields:
  title: 2.0
"""

MINI_LAYER = """
k1: 1.2
fields:
  title: 3.0
  abstract: 1.0
"""


@pytest.fixture
def tmp_repo(tmp_path) -> Path:
    """Một repo tí hon có `configs/base.yaml` + `configs/bm25.yaml`."""
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "base.yaml").write_text(MINI_BASE, encoding="utf-8")
    (configs / "bm25.yaml").write_text(MINI_LAYER, encoding="utf-8")
    return tmp_path


class TestLoadConfig:
    def test_base_co_du_ba_khoi(self):
        cfg = load_config("base")
        assert set(cfg) >= {"paths", "competition", "fields", "runtime"}

    def test_base_khong_nap_tang_khac(self):
        # Nếu base tự merge bm25 thì mọi lần chạy dense cũng dính k1/b.
        assert "implementation" not in load_config("base")

    def test_tang_bm25_merge_len_base(self):
        cfg = load_config("bm25")
        assert cfg["k1"] == 0.9
        assert cfg["b"] == 0.4
        assert cfg["paths"]["doc_store"] == "data/processed/doc_store.parquet"

    def test_tang_sau_ghi_de_tang_truoc(self, tmp_repo):
        cfg = load_config("bm25", repo_root=tmp_repo)
        assert cfg["k1"] == 1.2
        assert cfg["fields"]["title"] == 3.0
        assert cfg["fields"]["abstract"] == 1.0

    def test_override_ghi_de_tang_cuoi(self):
        cfg = load_config("bm25", overrides={"k1": 1.5, "fields": {"title": 4.0}})
        assert cfg["k1"] == 1.5
        # Merge theo từng khoá: đè fields.title nhưng giữ fields.abstract
        assert cfg["fields"]["title"] == 4.0
        assert cfg["fields"]["abstract"] == 1.0

    def test_override_none_bi_bo_qua(self):
        cfg = load_config("bm25", overrides={"k1": None})
        assert cfg["k1"] == 0.9

    def test_tang_khong_ton_tai_thi_raise(self):
        with pytest.raises(FileNotFoundError, match="hấy file cấu hình"):
            load_config("khong_co_tang_nay")

    def test_thieu_base_yaml_thi_raise(self, tmp_path):
        (tmp_path / "configs").mkdir()
        with pytest.raises(FileNotFoundError, match="cấu hình gốc"):
            load_config("base", repo_root=tmp_path)

    def test_ghi_lai_repo_root(self, tmp_repo):
        cfg = load_config("base", repo_root=tmp_repo)
        assert cfg["_meta"]["repo_root"] == str(tmp_repo)

    def test_co_query_field_trong_competition(self):
        assert load_config("base")["competition"]["query_field"] in {
            "query",
            "question",
            "narrative",
        }


class TestResolvePath:
    def test_duong_dan_tuong_doi_thanh_tuyet_doi(self, tmp_repo):
        cfg = load_config("base", repo_root=tmp_repo)
        assert resolve_path(cfg, "doc_store") == tmp_repo / "data/processed/doc_store.parquet"

    def test_duong_dan_tuyet_doi_giu_nguyen(self, tmp_repo):
        cfg = load_config("base", repo_root=tmp_repo)
        cfg["paths"]["doc_store"] = str(tmp_repo / "x.parquet")
        assert resolve_path(cfg, "doc_store") == tmp_repo / "x.parquet"

    def test_khoa_la_thi_raise(self):
        with pytest.raises(KeyError, match="paths"):
            resolve_path(load_config("base"), "khong_co_khoa")

    def test_resolve_any_path_competition(self, tmp_repo):
        cfg = load_config("base", repo_root=tmp_repo)
        assert resolve_any_path(cfg, "topics", "competition") == (
            tmp_repo / "data/external/topics-rnd3.csv"
        )

    def test_resolve_any_path_section_la_thi_raise(self):
        with pytest.raises(KeyError):
            resolve_any_path(load_config("base"), "topics", "khong_co_section")


class TestDeepMerge:
    def test_gop_dict_long_nhau(self):
        assert _deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}}) == {"a": {"b": 1, "c": 3}}

    def test_khong_sua_dict_goc(self):
        base = {"a": {"b": 1}}
        _deep_merge(base, {"a": {"b": 2}})
        assert base == {"a": {"b": 1}}

    def test_gia_tri_khong_phai_dict_thi_de(self):
        assert _deep_merge({"a": 1}, {"a": {"b": 2}}) == {"a": {"b": 2}}

    def test_them_khoa_moi(self):
        assert _deep_merge({}, {"a": 1}) == {"a": 1}

    def test_path_duoc_giu_kieu(self):
        out = _deep_merge({}, {"p": Path("x")})
        assert isinstance(out["p"], Path)
