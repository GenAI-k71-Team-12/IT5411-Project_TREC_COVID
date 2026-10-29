"""Test nhánh từ khoá: dựng chỉ mục BM25 và truy vấn.

Chạy trên backend `bm25s` (thuần Python) để CI không cần Java. Nhánh Lucene
dùng chung lớp `Bm25Retriever` nên phần lớn bất biến ở đây vẫn đúng.
"""

from __future__ import annotations

import json

import pytest

from trec_covid.index.bm25_index import (
    CORPUS_FILE,
    META_FILE,
    build_corpus,
    build_index,
    detect_backend,
    index_dir_for,
    load_index_meta,
)
from trec_covid.retrieve.bm25 import Bm25Retriever, search_bm25
from trec_covid.schemas import ScoredDocument

# ---------------------------------------------------------------------------
# Chọn backend
# ---------------------------------------------------------------------------


class TestDetectBackend:
    def test_ep_bm25s_thi_luon_dung(self):
        assert detect_backend("bm25s") == "bm25s"

    def test_khong_truyen_thi_mac_dinh_pyserini_hoac_lui(self):
        # Không raise trong mọi trường hợp môi trường.
        assert detect_backend(None) in {"pyserini", "bm25s"}

    def test_backend_la_thi_raise(self):
        with pytest.raises(ValueError, match="không hợp lệ"):
            detect_backend("elasticsearch")


# ---------------------------------------------------------------------------
# Dựng corpus
# ---------------------------------------------------------------------------


class TestBuildCorpus:
    def test_dung_thu_tu_doc_store(self, mini_doc_store):
        docs = build_corpus(mini_doc_store)
        assert [d["id"] for d in docs] == mini_doc_store["cord_uid"].tolist()

    def test_id_la_chuoi(self, mini_doc_store):
        assert all(isinstance(d["id"], str) for d in build_corpus(mini_doc_store))

    def test_title_duoc_lap_theo_trong_so(self, mini_doc_store):
        docs = {d["id"]: d["contents"] for d in build_corpus(mini_doc_store, title_weight=2)}
        title = mini_doc_store.set_index("cord_uid").loc["zzz99999", "title"]
        assert docs["zzz99999"].count(title) == 2

    def test_bai_chi_co_title_khong_bi_lap(self, mini_doc_store):
        docs = {d["id"]: d["contents"] for d in build_corpus(mini_doc_store, title_weight=2)}
        assert docs["a1b2c3d4"] == mini_doc_store.set_index("cord_uid").loc["a1b2c3d4", "title"]


# ---------------------------------------------------------------------------
# Dựng chỉ mục
# ---------------------------------------------------------------------------


class TestBuildIndex:
    def test_tao_meta_va_corpus(self, mini_index_dir):
        assert (mini_index_dir / META_FILE).is_file()
        assert (mini_index_dir / CORPUS_FILE).is_file()

    def test_meta_dung_thong_so(self, mini_index_dir, mini_doc_store):
        meta = load_index_meta(mini_index_dir)
        assert meta["backend"] == "bm25s"
        assert meta["n_docs"] == len(mini_doc_store)
        assert meta["k1"] == 0.9
        assert meta["b"] == 0.4
        assert len(meta["corpus_sha256"]) == 64

    def test_corpus_jsonl_dung_so_dong_va_thu_tu(self, mini_index_dir, mini_doc_store):
        lines = (mini_index_dir / CORPUS_FILE).read_text(encoding="utf-8").splitlines()
        assert len(lines) == len(mini_doc_store)
        ids = [json.loads(line)["id"] for line in lines]
        assert ids == mini_doc_store["cord_uid"].tolist()

    def test_index_dir_for_tro_dung_thu_muc_con(self, mini_index_dir):
        meta = load_index_meta(mini_index_dir)
        assert index_dir_for(mini_index_dir, meta).name == "bm25s"

    def test_goi_lai_khong_dung_lai(self, mini_index_dir, mini_doc_store):
        first = load_index_meta(mini_index_dir)
        second = build_index(mini_doc_store, mini_index_dir, backend="bm25s")
        assert second["created_at"] == first["created_at"]

    def test_force_thi_dung_lai(self, mini_index_dir, mini_doc_store):
        old = load_index_meta(mini_index_dir)["corpus_sha256"]
        build_index(mini_doc_store, mini_index_dir, backend="bm25s", title_weight=1.0, force=True)
        new = load_index_meta(mini_index_dir)
        assert new["title_weight"] == 1.0
        assert new["corpus_sha256"] != old

    def test_doc_store_rong_thi_raise(self, tmp_path, mini_doc_store):
        empty = mini_doc_store.iloc[0:0]
        with pytest.raises(ValueError, match="rỗng"):
            build_index(empty, tmp_path / "idx", backend="bm25s")

    def test_chua_co_meta_thi_raise(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="meta.json"):
            load_index_meta(tmp_path / "khong_co")


# ---------------------------------------------------------------------------
# Truy vấn
# ---------------------------------------------------------------------------


class TestBm25Retriever:
    def test_nap_duoc_chi_muc(self, mini_index_dir):
        r = Bm25Retriever(mini_index_dir)
        assert len(r) == 5
        assert r.backend == "bm25s"

    def test_thieu_chi_muc_thi_raise(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="02_build_bm25_index"):
            Bm25Retriever(tmp_path / "chua_dung")

    def test_tra_ve_scored_document(self, mini_index_dir):
        hits = Bm25Retriever(mini_index_dir).search("dexamethasone mortality", top_k=3)
        assert hits
        assert all(isinstance(h, ScoredDocument) for h in hits)
        assert all(h.source == "bm25" for h in hits)

    def test_rank_bat_dau_tu_1(self, mini_index_dir):
        hits = Bm25Retriever(mini_index_dir).search("dexamethasone mortality", top_k=5)
        assert [h.rank for h in hits] == list(range(1, len(hits) + 1))

    def test_score_giam_dan(self, mini_index_dir):
        scores = [h.score for h in Bm25Retriever(mini_index_dir).search("remdesivir", top_k=5)]
        assert scores == sorted(scores, reverse=True)

    def test_tim_dung_bai(self, mini_index_dir):
        hits = Bm25Retriever(mini_index_dir).search("dexamethasone", top_k=3)
        assert hits[0].cord_uid == "zzz99999"

    def test_truy_van_rong_tra_ve_danh_sach_rong(self, mini_index_dir):
        r = Bm25Retriever(mini_index_dir)
        assert r.search("") == []
        assert r.search("   ") == []

    def test_truy_van_toan_stopword_tra_ve_rong(self, mini_index_dir):
        # Không có token nội dung -> không được để bm25s trả kết quả ngẫu nhiên.
        assert Bm25Retriever(mini_index_dir).search("the and of") == []

    def test_top_k_lon_hon_so_bai(self, mini_index_dir):
        hits = Bm25Retriever(mini_index_dir).search("patients", top_k=1000)
        assert len(hits) == 5

    def test_top_k_nho(self, mini_index_dir):
        assert len(Bm25Retriever(mini_index_dir).search("patients", top_k=2)) == 2

    def test_khong_trung_cord_uid(self, mini_index_dir):
        hits = Bm25Retriever(mini_index_dir).search("patients", top_k=5)
        assert len({h.cord_uid for h in hits}) == len(hits)

    def test_moi_cord_uid_co_trong_doc_store(self, mini_index_dir, mini_doc_store):
        hits = Bm25Retriever(mini_index_dir).search("patients", top_k=5)
        assert {h.cord_uid for h in hits} <= set(mini_doc_store["cord_uid"])

    def test_query_co_ky_tu_lucene_khong_lam_sap(self, mini_index_dir):
        # Dấu ngoặc/ký tự đặc biệt từng làm crash Lucene QueryParser.
        hits = Bm25Retriever(mini_index_dir).search('COVID-19 (severe) "remdesivir"', top_k=5)
        assert isinstance(hits, list)

    def test_query_co_nbsp_van_tim_duoc(self, mini_index_dir):
        assert Bm25Retriever(mini_index_dir).search("COVID\u00a019", top_k=3)


class TestSearchBm25TienIch:
    def test_dung_index_dir_truyen_vao(self, mini_index_dir):
        hits = search_bm25("dexamethasone", top_k=3, index_dir=mini_index_dir)
        assert hits[0].cord_uid == "zzz99999"

    def test_cache_dung_lai_retriever(self, mini_index_dir):
        import trec_covid.retrieve.bm25 as mod

        mod._CACHE.clear()
        search_bm25("patients", top_k=1, index_dir=mini_index_dir)
        search_bm25("mortality", top_k=1, index_dir=mini_index_dir)
        assert len(mod._CACHE) == 1
