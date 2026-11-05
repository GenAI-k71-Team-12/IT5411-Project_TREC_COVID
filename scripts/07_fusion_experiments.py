"""So sánh 3 công thức dung hợp với 2 baseline đơn nhánh, tuning bằng 5-fold CV theo topic.

Đọc hai run nhánh (bàn giao của Người 1 và Người 2), quét lưới:
    weighted_rrf: w_bm25 ∈ {0.0, 0.1, ..., 1.0}, w_dense = 1 - w_bm25, k = 60
    convex:       alpha  ∈ {0.0, 0.1, ..., 1.0}
RRF chuẩn giữ k = 60, không tune.

Độ đo để chọn tham số (`--metric`, ADR 0010):
    condensed  nDCG@10 trên run đã bỏ bài chưa chấm (mặc định; công bằng khi qrels thưa)
    raw        nDCG@10 thường (thiên vị nhánh giống các run đã góp vào pool vòng 1-2)
Bảng luôn báo cả hai.

Ghi:
    runs/rrf.txt, runs/weighted_rrf.txt, runs/convex.txt   (tham số tốt nhất trên cả 35 topic)
    results/<tag>_fusion-comparison.csv                    (bảng 5 dòng)
    results/<tag>_fusion-grid.csv                          (đường điểm theo tham số)

    python scripts/07_fusion_experiments.py --bm25 runs/bm25.txt --dense runs/dense.txt
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # noqa: E402

import pandas as pd  # noqa: E402

from trec_covid.config import load_config, resolve_any_path, resolve_path  # noqa: E402
from trec_covid.evaluate.cv import cross_validate  # noqa: E402
from trec_covid.evaluate.results import write_results_csv  # noqa: E402
from trec_covid.evaluate.run_writer import Run, read_run, write_run  # noqa: E402
from trec_covid.evaluate.trec_eval import (  # noqa: E402
    condense,
    evaluate_run,
    judged_at_k,
    load_qrels,
    summarize,
)
from trec_covid.fuse.rrf import convex, drop_unmatched, rrf, weighted_rrf  # noqa: E402

GRID = [round(i / 10, 1) for i in range(11)]
TOP_K = int(load_config("fusion")["top_k"])  # đầu ra top-1000 theo hợp đồng


def fuse_all(bm25: Run, dense: Run, fn) -> Run:
    """Áp `fn(bm25_docs, dense_docs)` cho mọi topic có ở ít nhất một nhánh, cắt còn TOP_K."""
    topics = sorted(bm25.keys() | dense.keys(), key=int)
    return {t: fn(drop_unmatched(bm25.get(t, [])), dense.get(t, []))[:TOP_K] for t in topics}


def ndcg10(run: Run, qrels, metric: str) -> pd.Series:
    """nDCG@10 từng topic theo độ đo chọn (`condensed` hoặc `raw`)."""
    scored = condense(run, qrels) if metric == "condensed" else run
    return evaluate_run(scored, qrels)["ndcg_cut_10"]


def sweep(bm25: Run, dense: Run, qrels, make_fn, metric: str) -> dict[float, dict[str, float]]:
    """Điểm từng topic cho mỗi giá trị trong GRID. Chỉ fuse các topic có qrels."""
    b = {t: d for t, d in bm25.items() if t in qrels}
    d = {t: x for t, x in dense.items() if t in qrels}
    return {p: ndcg10(fuse_all(b, d, make_fn(p)), qrels, metric).to_dict() for p in GRID}


def row(name: str, run: Run, qrels, metric: str, cv: float, param: str) -> dict:
    sel = ndcg10(run, qrels, metric)
    return {
        "run": name,
        **summarize(evaluate_run(run, qrels)),
        "ndcg10_condensed": float(ndcg10(run, qrels, "condensed").mean()),
        "judged_10": float(judged_at_k(run, qrels).mean()),
        "ndcg10_cv5": cv,
        "param": param,
        "_sel": sel,
    }


def add_vs_best_single(rows: list[dict]) -> None:
    """Δ và số topic thắng/thua so với nhánh đơn tốt nhất, theo độ đo chọn."""
    single = max(rows[:2], key=lambda r: r["_sel"].mean())
    for r in rows:
        diff = r["_sel"] - single["_sel"]
        r["delta_vs_best_single"] = float(diff.mean())
        r["wins"], r["losses"] = int((diff > 1e-9).sum()), int((diff < -1e-9).sum())
        r["best_single"] = single["run"]


def grid_table(grids: dict[str, dict[float, dict[str, float]]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"formula": name, "param": p, "ndcg10_selected": pd.Series(s).mean()}
            for name, grid in grids.items()
            for p, s in grid.items()
        ]
    )


def config_drift(cfg, w_bm25: float, alpha: float, active: str) -> list[str]:
    """Các tham số trong configs/fusion.yaml lệch với kết quả tuning vừa chạy.

    04 đọc tham số từ config còn 07 ghi run bằng tham số vừa tune: lệch mà không báo thì
    bảng ablation trộn hai giá trị alpha khác nhau.
    """
    pairs = [
        ("active", cfg["active"], active),
        ("weighted_rrf.w_bm25", float(cfg["weighted_rrf"]["w_bm25"]), w_bm25),
        ("convex.alpha", float(cfg["convex"]["alpha"]), alpha),
    ]
    return [f"{k}: config={old} != tuning={new}" for k, old, new in pairs if old != new]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Thí nghiệm 3 công thức dung hợp")
    p.add_argument("--bm25", type=Path, default=Path("runs/bm25.txt"))
    p.add_argument("--dense", type=Path, default=Path("runs/dense.txt"))
    p.add_argument("--metric", choices=["condensed", "raw"], default="condensed")
    p.add_argument("--tag", default=date.today().isoformat(), help="tiền tố tên file kết quả")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config("fusion")
    qrels = load_qrels(resolve_any_path(cfg, "qrels", "competition"))
    bm25, dense = read_run(args.bm25), read_run(args.dense)
    k, m = int(cfg["rrf"]["k"]), args.metric

    w_grid = sweep(bm25, dense, qrels, lambda w: lambda b, d: weighted_rrf(b, d, w, 1 - w, k), m)
    a_grid = sweep(bm25, dense, qrels, lambda a: lambda b, d: convex(b, d, a), m)
    w_cv, a_cv = cross_validate(w_grid), cross_validate(a_grid)
    w, a = w_cv.best_param_all, a_cv.best_param_all

    fused = {
        "rrf": fuse_all(bm25, dense, lambda b, d: rrf(b, d, k)),
        "weighted_rrf": fuse_all(bm25, dense, lambda b, d: weighted_rrf(b, d, w, 1 - w, k)),
        "convex": fuse_all(bm25, dense, lambda b, d: convex(b, d, a)),
    }
    for name, run in fused.items():
        write_run(run, resolve_path(cfg, "runs") / f"{name}.txt", name)

    nan = float("nan")
    rrf_cv = float(ndcg10(fused["rrf"], qrels, m).mean())  # không tune -> CV = mean
    rows = [
        row("bm25", bm25, qrels, m, nan, ""),
        row("dense", dense, qrels, m, nan, ""),
        row("rrf", fused["rrf"], qrels, m, rrf_cv, f"k={k}"),
        row("weighted_rrf", fused["weighted_rrf"], qrels, m, w_cv.cv_mean,
            f"w_bm25={w} folds={w_cv.fold_params}"),
        row("convex", fused["convex"], qrels, m, a_cv.cv_mean,
            f"alpha={a} folds={a_cv.fold_params}"),
    ]  # fmt: skip
    add_vs_best_single(rows)
    table = pd.DataFrame([{c: v for c, v in r.items() if c != "_sel"} for r in rows])

    tags = sorted({next(iter(r.values()))[0].source for r in (bm25, dense)})
    prov = {"metric": m, "cv": "5fold-seed42", "tags": ";".join(tags), "config": "fusion"}
    out = resolve_path(cfg, "results")
    write_results_csv(table, out / f"{args.tag}_fusion-comparison.csv", prov)
    grids = {"weighted_rrf(w_bm25)": w_grid, "convex(alpha)": a_grid}
    write_results_csv(grid_table(grids), out / f"{args.tag}_fusion-grid.csv", prov)
    cols = ["run", "ndcg_cut_10", "ndcg10_condensed", "ndcg10_cv5", "bpref", "judged_10",
            "wins", "losses", "param"]  # fmt: skip
    print(table[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    cv_of = {"rrf": rrf_cv, "weighted_rrf": w_cv.cv_mean, "convex": a_cv.cv_mean}
    drift = config_drift(cfg, w, a, max(cv_of, key=cv_of.get))
    for d in drift:
        print(f"[fusion] WARNING config lech ket qua tuning: {d}")
    if drift:
        print("[fusion] Sua configs/fusion.yaml theo dong tren TRUOC khi chay 04/08.")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
