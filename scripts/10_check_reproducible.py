"""Kiểm tra tái lập kết quả giữa hai máy (task 1.6.5).

Máy 1 (đã có kết quả):
    python scripts/10_check_reproducible.py --write results/manifest.json
    -> commit results/manifest.json

Máy 2 (clone sạch, đặt dữ liệu vào data/, chạy lại pipeline):
    python scripts/02_build_bm25_index.py --backend bm25s
    (sinh runs/bm25.txt, runs/dense.txt theo hướng dẫn của Người 1, Người 2)
    python scripts/07_fusion_experiments.py
    (07 thoát mã 3 nếu tham số tune lệch configs/fusion.yaml: sửa config rồi chạy tiếp)
    python scripts/04_make_run.py --out runs/rerank.txt
    python scripts/04_make_run.py --from-run runs/rerank.txt --postfilter docids --out runs/rerank_docids.txt
    python scripts/04_make_run.py --from-run runs/rerank.txt --postfilter judged --out runs/rerank_judged.txt
    python scripts/10_check_reproducible.py --write results/manifest_may2.json \
        --compare results/manifest.json

Khớp nghĩa là: mọi độ đo khớp 4 chữ số và thứ tự top-10 từng topic khớp tuyệt đối
(không băm điểm vì fp16 trên GPU khác có thể lệch chữ số cuối).
"""  # noqa: E501

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # noqa: E402

from trec_covid.config import load_config, resolve_any_path  # noqa: E402
from trec_covid.evaluate.reproduce import compare_manifests, run_manifest  # noqa: E402
from trec_covid.evaluate.trec_eval import load_qrels  # noqa: E402


def default_runs(fusion_active: str) -> dict[str, Path]:
    names = ["bm25", "dense", fusion_active, "rerank", "rerank_docids", "rerank_judged"]
    return {n: Path("runs") / f"{n}.txt" for n in names}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Manifest tái lập + so sánh giữa hai máy")
    p.add_argument("--write", type=Path, required=True, help="ghi manifest của máy này")
    p.add_argument("--compare", type=Path, default=None, help="manifest của máy kia")
    return p.parse_args(argv)


def utf8_console() -> None:
    """In tiếng Việt an toàn khi stdout bị redirect trên Windows (mặc định cp1252)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    utf8_console()
    args = parse_args(argv)
    cfg = load_config("fusion")
    qrels = load_qrels(resolve_any_path(cfg, "qrels", "competition"))
    manifest = run_manifest(default_runs(cfg["active"]), qrels)
    args.write.parent.mkdir(parents=True, exist_ok=True)
    args.write.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[repro] Ghi {args.write} ({len(manifest['runs'])} run)")

    if args.compare is None:
        return 0
    other = json.loads(args.compare.read_text(encoding="utf-8"))
    problems = compare_manifests(other, manifest)
    for p in problems:
        print(f"[repro] KHÁC: {p}")
    print("[repro] TÁI LẬP ĐƯỢC" if not problems else f"[repro] {len(problems)} khác biệt")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
