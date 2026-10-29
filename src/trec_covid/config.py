"""Đọc và hợp nhất cấu hình YAML.

Thứ tự merge:

    configs/base.yaml -> configs/<tầng>.yaml -> tham số truyền lúc chạy

Tham số sau ghi đè tham số trước. Không hard-code tham số trong code.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

# src/trec_covid/config.py -> parents[2] là gốc repo
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge đệ quy: giá trị trong `override` thắng, dict thì gộp từng khoá."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(dict(out[key]), value)
        else:
            out[key] = value
    return out


def load_config(
    name: str = "base",
    *,
    overrides: Mapping[str, Any] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Nạp `configs/base.yaml` rồi merge `configs/<name>.yaml`.

    Args:
        name: tên tầng cấu hình (`bm25`, `dense`, `fusion`, ...). `"base"` thì chỉ nạp base.
        overrides: dict ghi đè cuối cùng (thường lấy từ argparse).
        repo_root: gốc repo, mặc định là thư mục chứa package.

    Returns:
        dict cấu hình đã merge.
    """
    root = Path(repo_root) if repo_root else REPO_ROOT
    config_dir = root / "configs"

    base_path = config_dir / "base.yaml"
    if not base_path.is_file():
        raise FileNotFoundError(f"Không thấy file cấu hình gốc: {base_path}")
    with base_path.open(encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh) or {}

    if name and name != "base":
        layer_path = config_dir / f"{name}.yaml"
        if not layer_path.is_file():
            raise FileNotFoundError(f"Không thấy file cấu hình: {layer_path}")
        with layer_path.open(encoding="utf-8") as fh:
            layer = yaml.safe_load(fh) or {}
        cfg = _deep_merge(cfg, layer)

    if overrides:
        cfg = _deep_merge(cfg, {k: v for k, v in overrides.items() if v is not None})

    cfg.setdefault("_meta", {})["repo_root"] = str(root)
    cfg["_meta"]["config_name"] = name
    return cfg


def resolve_path(cfg: Mapping[str, Any], key: str) -> Path:
    """Đổi một khoá trong `paths:` thành đường dẫn tuyệt đối.

    >>> resolve_path(cfg, "doc_store")   # doctest: +SKIP
    PosixPath('/.../data/processed/doc_store.parquet')
    """
    paths = cfg.get("paths", {})
    if key not in paths:
        raise KeyError(f"Không có khoá paths.{key} trong cấu hình")
    raw = Path(str(paths[key]))
    if raw.is_absolute():
        return raw
    root = Path(cfg.get("_meta", {}).get("repo_root", REPO_ROOT))
    return root / raw


def resolve_any_path(cfg: Mapping[str, Any], key: str, section: str) -> Path:
    """Như `resolve_path` nhưng tra trong một section khác (`competition`, ...)."""
    block = cfg.get(section, {})
    if key not in block:
        raise KeyError(f"Không có khoá {section}.{key} trong cấu hình")
    raw = Path(str(block[key]))
    if raw.is_absolute():
        return raw
    root = Path(cfg.get("_meta", {}).get("repo_root", REPO_ROOT))
    return root / raw
