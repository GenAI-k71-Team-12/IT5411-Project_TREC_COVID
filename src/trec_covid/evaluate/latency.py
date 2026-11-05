"""Đo latency từng tầng: median và p95 trên một tập truy vấn, bỏ các lần chạy khởi động."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd


def time_stages(
    stages: Mapping[str, Callable[[str], object]],
    queries: Sequence[str],
    warmup: int = 3,
) -> pd.DataFrame:
    """Gọi từng `stages[name](query)`; `warmup` lần đầu không tính (nạp model, JIT, cache).

    Returns:
        DataFrame cột `stage`, `median_ms`, `p95_ms`, `n`.
    """
    rows = []
    for name, fn in stages.items():
        for q in list(queries)[:warmup]:
            fn(q)
        times = []
        for q in queries:
            start = time.perf_counter()
            fn(q)
            times.append((time.perf_counter() - start) * 1000)
        rows.append(
            {
                "stage": name,
                "median_ms": float(np.median(times)),
                "p95_ms": float(np.percentile(times, 95)),
                "n": len(times),
            }
        )
    return pd.DataFrame(rows)
