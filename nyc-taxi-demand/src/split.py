"""
split.py — Phase 4: 時系列日付分割

リーク防止方針:
  - 分割は必ず date 列の値で行う（ランダム分割禁止）。
  - train: date <= train_end
  - val:   train_end < date <= val_end
  - test:  date > val_end
  - NaN 行（ラグ/rolling 由来）は load_splits 内で dropna する。
    全条件で同じ行（ゾーン先頭最大14行相当）が落ちるため、
    条件間の対称性が保たれる。
  - 日付の大小比較のみで分割する。ランダムシャッフルを伴う関数は使用しない。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import load_config


def load_splits(
    condition: str = "A",
    cfg: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    data/features/dataset_{condition}.parquet を読み込み、
    日付による厳密な時系列分割を行い (train, val, test) を返す。

    Parameters
    ----------
    condition : str
        使用するアブレーション条件 ("A" / "B" / "C" / "D")。
    cfg : dict, optional
        設定辞書。省略時は config/config.yaml から読み込む。

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train, val, test) — それぞれ date/zone/demand + 特徴量列を保持。

    分割境界:
        train : date <= split.train_end  (2023-12-31)
        val   : split.train_end < date <= split.val_end  (2024-06-30)
        test  : date > split.val_end  (2024-07-01 〜)

    リーク防止:
        日付で厳密に切るため、未来データが train に混入しない。
        NaN 行を dropna することで StandardScaler.fit が安全に動作する。
        全条件で同じ NaN 行（ゾーン先頭のラグ由来 NaN）が落ちるため対称。
    """
    if cfg is None:
        cfg = load_config()

    # --- パス解決 ---
    project_root = Path(__file__).resolve().parent.parent
    features_dir = Path(cfg["paths"]["features_dir"])
    if not features_dir.is_absolute():
        features_dir = project_root / features_dir

    parquet_path = features_dir / f"dataset_{condition}.parquet"
    df = pd.read_parquet(parquet_path)

    # --- date 列を datetime 型に統一 ---
    df["date"] = pd.to_datetime(df["date"])

    # --- ラグ/rolling 由来の NaN を除去 ---
    # groupby('zone').shift(lag) で各ゾーン先頭 lag 行が NaN になる。
    # 全条件で同じ行に NaN が存在するため、dropna は条件間で対称的に働く。
    # train に NaN があると StandardScaler.fit が失敗するため、事前に除去する。
    before = len(df)
    df = df.dropna()
    after = len(df)
    dropped = before - after
    if dropped > 0:
        pass  # NaN 除去は正常動作（ラグ/rolling の先頭行）

    # --- 分割境界の取得（ハードコードせず config から読む）---
    train_end = pd.Timestamp(cfg["split"]["train_end"])
    val_end = pd.Timestamp(cfg["split"]["val_end"])

    # --- 日付による厳密な時系列分割 ---
    # リーク防止: date の大小比較のみで分割する。
    # ランダム要素は一切含まない。
    # train_end 以前が train, val_end 以前（かつ train_end 超過）が val,
    # それ以降が test となる。各区間は互いに重複しない。
    train = df[df["date"] <= train_end].copy()
    val = df[(df["date"] > train_end) & (df["date"] <= val_end)].copy()
    test = df[df["date"] > val_end].copy()

    # --- 不変条件のアサーション ---
    assert len(train) > 0, "train が空です"
    assert len(val) > 0, "val が空です"
    assert len(test) > 0, "test が空です"
    # 分割が重複しないことを確認（検査C の契約）
    assert train["date"].max() < val["date"].min(), "Train と Val の日付が重複"
    assert val["date"].max() < test["date"].min(), "Val と Test の日付が重複"

    return train, val, test


if __name__ == "__main__":
    for cond in ("A", "B", "C", "D"):
        tr, va, te = load_splits(cond)
        print(
            f"[{cond}] train={len(tr):,} val={len(va):,} test={len(te):,} "
            f"| train_end={tr['date'].max().date()} "
            f"| val_start={va['date'].min().date()} val_end={va['date'].max().date()} "
            f"| test_start={te['date'].min().date()} test_end={te['date'].max().date()}"
        )
