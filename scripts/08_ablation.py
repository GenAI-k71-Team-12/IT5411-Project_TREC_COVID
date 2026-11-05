"""Bảng ablation 5 tầng (+ một dòng đối chiếu) trên 35 topic có qrels.

Mặc định đọc các run đã sinh bởi 07 (fusion) và 04 (rerank, postfilter):

    BM25                 runs/bm25.txt
    Dense                runs/dense.txt
    Fusion               runs/<fusion.active>.txt
    +Rerank              runs/rerank.txt
    +Residual (docids)   runs/rerank_docids.txt     <- chế độ đúng thể lệ (ADR 0005)
    +Residual (judged)   runs/rerank_judged.txt     <- đối chiếu: xoá nhãn, ra 0

    python scripts/08_ablation.py
    python scripts/08_ablation.py --stage "BM25=runs/bm25.txt" --stage "Dense=runs/dense.txt"
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # noqa: E402

from trec_covid.config import load_config, resolve_any_path, resolve_path  # noqa: E402
from trec_covid.evaluate.ablation import ablation_table  # noqa: E402
from trec_covid.evaluate.results import write_results_csv  # noqa: E402
from trec_covid.evaluate.trec_eval import load_qrels  # noqa: E402


def default_stages(fusion_active: str) -> list[tuple[str, Path]]:
    runs = Path("runs")
    return [
        ("BM25", runs / "bm25.txt"),
        ("Dense", runs / "dense.txt"),
        (f"Fusion ({fusion_active})", runs / f"{fusion_active}.txt"),
        ("+Rerank", runs / "rerank.txt"),
        ("+Residual (docids)", runs / "rerank_docids.txt"),
        ("+Residual (judged)", runs / "rerank_judged.txt"),
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bảng ablation theo tầng")
    p.add_argument("--stage", action="append", default=None, help='"Tên=đường/dẫn/run.txt"')
    p.add_argument("--tag", default=date.today().isoformat(), help="tiền tố tên file kết quả")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    fusion, rerank = load_config("fusion"), load_config("rerank")
    stages = (
        [tuple(s.split("=", 1)) for s in args.stage]
        if args.stage
        else default_stages(fusion["active"])
    )
    table = ablation_table(stages, load_qrels(resolve_any_path(fusion, "qrels", "competition")))
    if not args.stage:
        # Tham số fusion là best-all (chọn trên chính 35 topic): lạc quan hơn số CV trung thực.
        table["note"] = ""
        table.loc[2:, "note"] = "tham số fusion in-sample; số CV ở fusion-comparison (ndcg10_cv5)"

    active = fusion["active"]
    params = {k: v for k, v in fusion.get(active, {}).items() if k != "minmax_scope"}
    prov = {
        "fusion": f"{active}{params}".replace(" ", ""),
        "rerank": f"{rerank['model']}@top{rerank['top_k']}",
        "postfilter_default": load_config("postfilter")["mode"],
    }
    out = resolve_path(fusion, "results") / f"{args.tag}_ablation.csv"
    write_results_csv(table, out, prov)
    cols = ["stage", "ndcg_cut_10", "delta_ndcg10", "ndcg10_condensed", "delta_condensed",
            "map", "P_5", "bpref", "judged_10"]  # fmt: skip
    print(table[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"[ablation] Ghi {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
