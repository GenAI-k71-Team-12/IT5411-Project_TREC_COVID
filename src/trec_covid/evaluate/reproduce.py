"""Manifest kết quả để kiểm tra tái lập giữa hai máy (task 1.6.5).

Không băm điểm số: rerank chạy fp16 trên GPU khác nhau có thể lệch ở chữ số cuối mà
không đổi thứ hạng. Manifest giữ:
    - độ đo làm tròn 4 chữ số (phải khớp tuyệt đối)
    - SHA256 thứ tự top-10 docid từng topic (phải khớp tuyệt đối)
    - SHA256 thứ tự toàn bộ run (chỉ để tham khảo: hoà điểm sâu trong danh sách có thể
      đảo thứ tự giữa hai GPU mà không đổi độ đo)
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from trec_covid.evaluate.run_writer import Run, read_run
from trec_covid.evaluate.trec_eval import evaluate_run, summarize


def _order_hash(run: Run, depth: int | None) -> str:
    h = hashlib.sha256()
    for topic in sorted(run, key=int):
        h.update(f"{topic}:{','.join(d.cord_uid for d in run[topic][:depth])}\n".encode())
    return h.hexdigest()


def run_manifest(
    runs: Mapping[str, str | Path], qrels: Mapping[str, Mapping[str, int]]
) -> dict[str, Any]:
    """`{"runs": {tên: {"metrics": {...}, "top10_sha256": ..., "order_sha256": ...}}}`."""
    out: dict[str, Any] = {"runs": {}}
    for name, path in runs.items():
        run = read_run(path)
        metrics = summarize(evaluate_run(run, qrels))
        out["runs"][name] = {
            "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
            "top10_sha256": _order_hash(run, 10),
            "order_sha256": _order_hash(run, None),
        }
    return out


def compare_manifests(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[str]:
    """Danh sách khác biệt bắt buộc phải khớp; rỗng nghĩa là tái lập được."""
    problems: list[str] = []
    for name, ra in a["runs"].items():
        rb = b["runs"].get(name)
        if rb is None:
            problems.append(f"thiếu run {name} ở manifest thứ hai")
            continue
        if ra["top10_sha256"] != rb["top10_sha256"]:
            problems.append(f"{name}: thứ tự top-10 khác nhau")
        for metric, va in ra["metrics"].items():
            vb = rb["metrics"].get(metric)
            if vb != va:
                problems.append(f"{name}: {metric} {va} != {vb}")
    return problems
