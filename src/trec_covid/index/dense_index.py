"""Encode corpus bằng MedCPT Article Encoder và nạp vào FAISS IndexFlatIP.

Các bước:
    1. Nạp Doc Store
    2. Encode (title, abstract) theo chunk, max_length = 512, lưu từng chunk .npy
    3. Chuẩn hoá L2 vector        <-- BẮT BUỘC, nếu không IP != cos
    4. Nạp vào IndexFlatIP (không có bước train)
    5. Ghi xuống đĩa: articles.faiss + articles.docids.txt + articles.meta.json

Kích thước: 170k x 768 x 4 byte ~ 500 MB.  Thời gian encode: ~2-3 giờ (GPU).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

DOCIDS_SUFFIX = ".docids.txt"
META_SUFFIX = ".meta.json"


# ---------------------------------------------------------------------------
# Tiện ích
# ---------------------------------------------------------------------------


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Chuẩn hoá L2 từng dòng, trả float32 liên tục (FAISS yêu cầu float32 C-contiguous)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def docids_sha256(docids: Sequence[str]) -> str:
    h = hashlib.sha256()
    for uid in docids:
        h.update(uid.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Nhiệm vụ 1 — Encoder
# ---------------------------------------------------------------------------

#: Với Article Encoder: (title, abstract). Với Query Encoder: một chuỗi.
EncoderInput = str | tuple[str, str]


class MedCPTEncoder:
    """Bọc một checkpoint MedCPT (Article hoặc Query). Nạp model MỘT lần."""

    def __init__(
        self,
        model_name: str,
        *,
        max_length: int,
        batch_size: int = 32,
        device: str = "auto",
        fp16: bool = True,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.model = model.half() if fp16 and self.device == "cuda" else model
        self._torch = torch

    def _embed_batch(self, batch: Sequence[EncoderInput]) -> np.ndarray:
        enc = self.tokenizer(
            [list(b) if isinstance(b, tuple) else b for b in batch],
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        with self._torch.no_grad():
            cls = self.model(**enc).last_hidden_state[:, 0, :]   # [CLS], không mean-pool
        return cls.float().cpu().numpy()

    def encode(self, items: Sequence[EncoderInput]) -> np.ndarray:
        """Trả ma trận (n, dim) float32 **chưa chuẩn hoá**, ĐÚNG thứ tự `items`."""
        n = len(items)
        if n == 0:
            return np.zeros((0, 768), dtype=np.float32)

        def length(x: EncoderInput) -> int:
            return sum(len(p) for p in x) if isinstance(x, tuple) else len(x)

        order = np.argsort([-length(x) for x in items], kind="stable")  # dài trước: ít padding
        out: np.ndarray | None = None
        for start in range(0, n, self.batch_size):
            idx = order[start : start + self.batch_size]
            vecs = self._embed_batch([items[i] for i in idx])
            if out is None:
                out = np.empty((n, vecs.shape[1]), dtype=np.float32)
            out[idx] = vecs                                      # trả về đúng chỉ số gốc
        assert out is not None
        return out


# ---------------------------------------------------------------------------
# Nhiệm vụ 2 — Encode corpus theo chunk (chạy tiếp được)
# ---------------------------------------------------------------------------


def encode_corpus(
    titles: Sequence[str],
    abstracts: Sequence[str],
    encoder: MedCPTEncoder,
    chunk_dir: str | Path,
    *,
    chunk_size: int = 4096,
    signature: dict[str, Any] | None = None,
) -> np.ndarray:
    """Encode toàn bộ corpus, mỗi chunk lưu `chunk_XXXXX.npy`; chunk đã có thì bỏ qua.

    Job chạy 2-3 giờ: mất điện / OOM giữa chừng thì chạy lại lệnh cũ, nó làm tiếp.
    Trả ma trận (n, dim) CHƯA chuẩn hoá, hàng i ứng với dòng i của Doc Store.

    `signature` (model, max_length, chunk_size, n...) được ghi vào `manifest.json`; chạy
    tiếp với signature khác thì raise, vì chunk cũ thuộc không gian vector khác.
    """
    from tqdm import tqdm

    assert len(titles) == len(abstracts)
    n = len(titles)
    out_dir = Path(chunk_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sig = {**(signature or {}), "chunk_size": chunk_size, "n": n}
    manifest = out_dir / "manifest.json"
    if manifest.is_file():
        old = json.loads(manifest.read_text(encoding="utf-8"))
        if old != sig:
            raise ValueError(
                f"Chunk cũ trong {out_dir} dựng bằng cấu hình khác ({old} != {sig}). "
                "Chạy lại với --force để encode lại từ đầu."
            )
    else:
        manifest.write_text(json.dumps(sig, indent=2, ensure_ascii=False), encoding="utf-8")

    parts: list[np.ndarray] = []
    starts = range(0, n, chunk_size)
    for ci, start in enumerate(tqdm(starts, desc="encode", unit="chunk")):
        end = min(start + chunk_size, n)
        path = out_dir / f"chunk_{ci:05d}.npy"
        if path.is_file():
            arr = np.load(path)
            if arr.shape[0] == end - start:        # chunk hỏng/ngắn -> encode lại
                parts.append(arr)
                continue
        items = [(titles[i], abstracts[i]) for i in range(start, end)]
        arr = encoder.encode(items)
        tmp = path.with_name(path.name + ".tmp.npy")
        np.save(tmp, arr)
        tmp.replace(path)                           # ghi nguyên tử: không để lại chunk nửa vời
        parts.append(arr)
    return np.concatenate(parts, axis=0) if parts else np.zeros((0, 768), dtype=np.float32)


# ---------------------------------------------------------------------------
# Nhiệm vụ 3 — Chuẩn hoá L2 + FAISS + ánh xạ position -> cord_uid
# ---------------------------------------------------------------------------


def build_faiss_index(embeddings: np.ndarray, docids: Sequence[str]):
    """Chuẩn hoá L2 rồi nạp vào `IndexFlatIP`. `index.ntotal` phải bằng `len(docids)`."""
    import faiss

    if embeddings.shape[0] != len(docids):
        raise ValueError(f"{embeddings.shape[0]} vector nhưng {len(docids)} cord_uid")
    if len(set(docids)) != len(docids):
        raise ValueError("cord_uid trùng trong docid_order — vị trí sẽ mơ hồ")
    vecs = l2_normalize(embeddings)
    index = faiss.IndexFlatIP(vecs.shape[1])        # không có bước train
    index.add(vecs)
    assert index.ntotal == len(docids)
    return index


def save_dense_index(
    index, docids: Sequence[str], path: str | Path, meta: dict[str, Any]
) -> Path:
    """Ghi `<path>` (faiss), `<stem>.docids.txt` (position -> cord_uid), `<stem>.meta.json`."""
    import faiss

    if index.ntotal != len(docids):
        raise ValueError(f"index.ntotal={index.ntotal} != len(docids)={len(docids)}")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out))
    out.with_suffix(DOCIDS_SUFFIX).write_text("\n".join(docids) + "\n", encoding="utf-8")
    full_meta = {
        **meta,
        "n_docs": len(docids),
        "dim": int(index.d),
        "normalized_l2": True,
        "docids_sha256": docids_sha256(docids),
    }
    out.with_suffix(META_SUFFIX).write_text(
        json.dumps(full_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out


def load_dense_index(path: str | Path) -> tuple[Any, list[str], dict[str, Any]]:
    """Nạp (index, docids, meta); raise nếu ánh xạ không khớp chỉ mục."""
    import faiss

    p = Path(path)
    for f in (p, p.with_suffix(DOCIDS_SUFFIX), p.with_suffix(META_SUFFIX)):
        if not f.is_file():
            raise FileNotFoundError(f"Thiếu {f} (chạy scripts/03_build_faiss_index.py)")
    index = faiss.read_index(str(p))
    docids = [ln for ln in p.with_suffix(DOCIDS_SUFFIX).read_text(encoding="utf-8").splitlines()]
    meta = json.loads(p.with_suffix(META_SUFFIX).read_text(encoding="utf-8"))
    if index.ntotal != len(docids):
        raise ValueError(f"index.ntotal={index.ntotal} != {len(docids)} cord_uid")
    if meta.get("docids_sha256") != docids_sha256(docids):
        raise ValueError("docids.txt không khớp meta.json — file đã bị sửa/trộn giữa hai lần dựng")
    return index, docids, meta
