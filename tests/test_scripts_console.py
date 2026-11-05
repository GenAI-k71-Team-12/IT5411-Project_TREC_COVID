"""Script phải chạy được khi stdout bị redirect/pipe trên Windows (bảng mã cp1252).

Không có test này, `print` tiếng Việt raise UnicodeEncodeError TRƯỚC khi ghi run file:
chạy rerank cả buổi rồi mất trắng kết quả chỉ vì `> log.txt`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_cp1252(args: list[str]) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONIOENCODING": "cp1252", "PYTHONUTF8": "0"}
    return subprocess.run(
        [sys.executable, *args], cwd=ROOT, env=env, capture_output=True, timeout=120
    )


def test_make_run_ghi_duoc_run_khi_stdout_la_cp1252(tmp_path):
    src = tmp_path / "in.txt"
    src.write_text("1 Q0 d1 1 2.000000 x\n1 Q0 d2 2 1.000000 x\n", encoding="utf-8")
    out = tmp_path / "out.txt"
    proc = _run_cp1252(
        ["scripts/04_make_run.py", "--from-run", str(src), "--postfilter", "off", "--out", str(out)]
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    assert out.read_text(encoding="utf-8").splitlines()[0] == "1 Q0 d1 1 2.000000 out"


def test_check_reproducible_khong_crash_va_bao_dung_khi_cp1252(tmp_path):
    # Cần run thật trong runs/ (không có trên máy chưa chạy pipeline) -> skip.
    import pytest

    if not (ROOT / "runs" / "rerank_judged.txt").is_file():
        pytest.skip("chưa có runs/ của pipeline")
    first = tmp_path / "a.json"
    proc = _run_cp1252(["scripts/10_check_reproducible.py", "--write", str(first)])
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    proc = _run_cp1252(
        [
            "scripts/10_check_reproducible.py",
            "--write",
            str(tmp_path / "b.json"),
            "--compare",
            str(first),
        ]
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
