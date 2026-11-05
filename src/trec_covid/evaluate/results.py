"""Ghi bảng kết quả vào `results/` kèm nguồn gốc (commit + config).

CONTRIBUTING §5: mỗi kết quả phải có dòng đầu ghi commit hash và config đã dùng,
nếu không thì không so sánh được giữa các lần chạy.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from trec_covid.config import REPO_ROOT


def git_commit() -> str:
    """Commit hiện tại (rút gọn), thêm `-dirty` nếu cây làm việc có thay đổi chưa commit."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{head}-dirty" if dirty else head


def write_results_csv(df: pd.DataFrame, path: str | Path, provenance: Mapping[str, str]) -> Path:
    """Ghi CSV, dòng đầu `# commit=<hash> k=v ...`. Đọc lại bằng `pd.read_csv(comment="#")`."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = " ".join(f"{k}={v}" for k, v in provenance.items())
    # Lấy commit TRƯỚC khi mở file: open("w") xoá nội dung file đã track, git sẽ báo dirty.
    header = f"# commit={git_commit()} {meta}\n"
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(header)
        df.to_csv(fh, index=False, float_format="%.4f", lineterminator="\n")
    return out
