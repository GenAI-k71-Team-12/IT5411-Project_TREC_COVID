"""Ghi và đọc run file 6 cột cho trec_eval.

    topicid Q0 docid rank score run-tag

LƯU Ý: trec_eval sắp xếp lại theo cột `score` và BỎ QUA cột `rank`.
Vì vậy score phải đơn điệu giảm theo thứ hạng cuối: `write_run` kiểm tra bằng
`check_ranking` và từ chối ghi nếu sai.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from trec_covid.fuse.ranking import check_ranking
from trec_covid.schemas import ScoredDocument

#: Run theo topic. Khoá là topic_id dạng chuỗi (giống qrels của trec_eval).
Run = dict[str, list[ScoredDocument]]


def write_run(
    run: Mapping[str | int, Sequence[ScoredDocument]],
    path: str | Path,
    run_tag: str,
) -> Path:
    """Ghi run file. Kiểm tra bất biến xếp hạng của từng topic trước khi ghi."""
    for docs in run.values():
        check_ranking(docs)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for topic_id, docs in run.items():
            for d in docs:
                fh.write(f"{topic_id} Q0 {d.cord_uid} {d.rank} {d.score:.6f} {run_tag}\n")
    return out


def read_run(path: str | Path) -> Run:
    """Đọc run file thành `{topic_id: [ScoredDocument]}`, giữ thứ tự dòng trong file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Không thấy run file: {p}")
    run: Run = {}
    with p.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 6:
                raise ValueError(f"{p.name}:{n} không đủ 6 cột: {line!r}")
            topic_id, _, uid, rank, score, tag = parts
            run.setdefault(topic_id, []).append(
                ScoredDocument(cord_uid=uid, score=float(score), rank=int(rank), source=tag)
            )
    return run
