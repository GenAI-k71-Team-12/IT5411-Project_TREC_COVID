"""Test nhánh dense bằng encoder giả (không cần torch/GPU/model thật)."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

pytest.importorskip("faiss")

from trec_covid.evaluate.run_writer import write_run  # noqa: E402
from trec_covid.fuse.ranking import check_ranking  # noqa: E402
from trec_covid.index.dense_index import (  # noqa: E402
    build_faiss_index,
    l2_normalize,
    load_dense_index,
    save_dense_index,
)
from trec_covid.retrieve.dense import DenseRetriever  # noqa: E402

DIM = 16


def fake_vec(text: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")
    vec = np.random.default_rng(seed).normal(size=DIM).astype(np.float32)
    return vec * 5  # cố ý không chuẩn hoá


class FakeEncoder:
    def encode(self, items):
        return np.stack([fake_vec(i if isinstance(i, str) else i[0]) for i in items])


@pytest.fixture
def index_path(tmp_path):
    docids = [f"doc{i}" for i in range(50)]
    vecs = np.stack([fake_vec(f"doc{i}") for i in range(50)])
    index = build_faiss_index(vecs, docids)
    path = tmp_path / "articles.faiss"
    save_dense_index(index, docids, path, {"query_encoder": "fake", "article_encoder": "fake"})
    return path


def test_l2_normalize_dua_ve_do_dai_1():
    out = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(np.linalg.norm(out[0]), 1.0)
    assert np.isfinite(out).all()                      # vector 0 không sinh NaN


def test_search_tra_dung_hop_dong(index_path):
    r = DenseRetriever(index_path, query_model="fake", encoder=FakeEncoder())
    res = r.search("doc7", top_k=10)
    assert [d.rank for d in res] == list(range(1, 11))
    assert all(d.source == "dense" for d in res)
    check_ranking(res)                                 # score không tăng, không trùng uid
    assert res[0].cord_uid == "doc7"                   # query == chính bài -> hạng 1
    assert res[0].score == pytest.approx(1.0, abs=1e-4)   # cosine với chính nó (đã chuẩn hoá L2)


def test_top_k_lon_hon_ntotal_va_query_rong(index_path):
    r = DenseRetriever(index_path, query_model="fake", encoder=FakeEncoder())
    assert len(r.search("doc1", top_k=1000)) == 50
    assert r.search("   ") == []
    assert r.search("doc1", top_k=0) == []


def test_mapping_lech_so_luong_bi_tu_choi():
    with pytest.raises(ValueError):
        build_faiss_index(np.zeros((3, DIM), dtype=np.float32), ["a", "b"])


def test_docids_bi_sua_thi_load_bao_loi(index_path):
    swapped = "\n".join(reversed([f"doc{i}" for i in range(50)]))
    index_path.with_suffix(".docids.txt").write_text(swapped)
    with pytest.raises(ValueError):
        load_dense_index(index_path)


def test_query_encoder_sai_checkpoint_bi_chan(index_path):
    with pytest.raises(ValueError):
        DenseRetriever(index_path, query_model="other-model")


def test_run_ghi_duoc(index_path, tmp_path):
    r = DenseRetriever(index_path, query_model="fake", encoder=FakeEncoder())
    out = write_run({"1": r.search("doc3", 20)}, tmp_path / "dense.txt", "dense")
    assert out.read_text().splitlines()[0].startswith("1 Q0 doc3 1 ")


def test_encode_tra_ve_dung_thu_tu_goc(monkeypatch):
    from trec_covid.index.dense_index import MedCPTEncoder

    enc = MedCPTEncoder.__new__(MedCPTEncoder)       # bỏ qua __init__ (không nạp model)
    enc.batch_size = 2
    monkeypatch.setattr(
        enc, "_embed_batch",
        lambda batch: np.array([[float(len(b))] for b in batch]),
    )
    items = ["a", "bbbb", "cc", "dddddd", "e"]       # độ dài lộn xộn
    out = enc.encode(items)
    assert out[:, 0].tolist() == [1.0, 4.0, 2.0, 6.0, 1.0]


def test_encode_corpus_chay_tiep_va_chan_khi_doi_cau_hinh(tmp_path):
    from trec_covid.index.dense_index import encode_corpus

    class Enc:
        calls = 0

        def encode(self, items):
            Enc.calls += len(items)
            return np.ones((len(items), 4), dtype=np.float32)

    titles, abstracts = list("abcde"), list("vwxyz")
    sig = {"model": "m", "max_length": 512}
    a = encode_corpus(titles, abstracts, Enc(), tmp_path, chunk_size=2, signature=sig)
    assert a.shape == (5, 4) and Enc.calls == 5
    encode_corpus(titles, abstracts, Enc(), tmp_path, chunk_size=2, signature=sig)
    assert Enc.calls == 5                              # chạy tiếp: không encode lại chunk nào
    with pytest.raises(ValueError):
        encode_corpus(titles, abstracts, Enc(), tmp_path, chunk_size=2,
                      signature={**sig, "model": "khac"})
