"""
config.py - config/config.yaml を読み込むヘルパー
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str = "config/config.yaml") -> dict[str, Any]:
    """
    config.yaml を読み込んで dict として返す。

    相対パスが渡された場合は、プロジェクトルート（このファイルの 2 つ上）を
    基準に解決する。
    """
    p = Path(path)
    if not p.is_absolute():
        # src/config.py -> src -> project_root
        project_root = Path(__file__).resolve().parent.parent
        p = project_root / p

    if not p.exists():
        raise FileNotFoundError(f"config file not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg


if __name__ == "__main__":
    cfg = load_config()
    import pprint
    pprint.pprint(cfg)
