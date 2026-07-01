"""
train.py — Phase 5-6: モデル学習・ハイパラ調整・アブレーション評価

リーク防止方針:
  1. 分割は src.split.load_splits の日付分割を使用（ランダム分割なし）。
  2. StandardScaler は train のみで fit し、val/test には transform のみ適用。
     線形系は Pipeline[StandardScaler → モデル] で内包する。
     木系（XGBoost）は scaler 不要のためそのまま使う。
  3. CV は TimeSeriesSplit のみ（KFold 禁止）。train 内のみで閉じる。
     パネルデータ（同一 date に 10 ゾーン）なので date 昇順でソートしてから CV。
  4. 全モデルで n_iter=cfg.tuning.n_iter を同一にし、探索予算を公平に揃える。
  5. val でモデル選択、test は最終報告のみ（1度だけ評価）。
  6. ゾーン別に MAE/RMSE/R² を計算してから集約（ゾーン平均）。
     MAPE は使わない（ゼロ需要で発散するため）。

zone の扱い方針（コメント）:
  zone は LocationID（整数）であり、順序のない名義変数。整数のまま渡すと
  「ゾーン ID の大小」という意味のない順序を学習し得るため、全モデルで
  one-hot（ダミー変数）化して投入する（build_zone_onehot_matrix）。
  ダミーのカテゴリは train から固定し、drop_first=True で k-1 個にして
  線形系の完全共線性（ダミートラップ）を避ける。SeasonalNaive は demand_lag_7
  のみを参照する基準モデルで特徴量行列を使わないため、この変更の影響を受けない。
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from scipy.stats import loguniform, randint, uniform

from src.config import load_config
from src.split import load_splits

logger = logging.getLogger(__name__)


# ===========================
# fit_scaler — ★テスト契約★
# ===========================

def fit_scaler(X: pd.DataFrame) -> StandardScaler:
    """
    StandardScaler を X（train のみ）でフィットして返す。

    リーク防止: この関数は train の DataFrame のみで呼び出すこと。
    val/test への適用は必ず transform のみとし、fit は絶対に行わない。
    検査D はこの関数が返す scaler.mean_ が train の平均と一致することを確認する。

    Parameters
    ----------
    X : pd.DataFrame
        フィット対象の特徴量 DataFrame（train のスライス）。

    Returns
    -------
    StandardScaler
        train のみでフィットされた StandardScaler インスタンス。
    """
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


# ===========================
# 名義変数（zone）の one-hot 化
# ===========================

def build_zone_onehot_matrix(
    df: pd.DataFrame,
    feature_cols: List[str],
    zone_categories: List,
    drop_first: bool = True,
) -> pd.DataFrame:
    """
    feature_cols（date/zone/demand を除く特徴量列）に、名義変数 zone を
    one-hot 化したダミー列を結合した特徴量行列 X を返す。

    zone（= LocationID）は順序を持たない名義変数なので、整数のまま渡さず
    ダミー変数化して全モデルで使用する。

    リーク防止 / 再現性:
      - zone のカテゴリ集合は train から決めた zone_categories に固定する。
        これにより val/test でも train と同一のダミー列構成（同じ順・同じ列数）に
        なり、仮に欠けたゾーンがあっても全ゼロ列として揃う（学習時と推論時で
        列がずれない）。カテゴリ決定に val/test を覗かない。
      - drop_first=True で k-1 個のダミーにし、基準ゾーンを切片に吸収させて
        線形モデルの完全共線性（ダミートラップ）を避ける。
    """
    X = df[feature_cols].copy()
    zone_cat = pd.Categorical(df["zone"], categories=zone_categories)
    dummies = pd.get_dummies(zone_cat, prefix="zone", drop_first=drop_first).astype(float)
    dummies.index = X.index
    return pd.concat([X, dummies], axis=1)


# ===========================
# 評価関数（ゾーン別 → 集約）
# ===========================

def evaluate_by_zone(
    df_eval: pd.DataFrame,
    y_pred: np.ndarray,
    split_name: str,
) -> Dict[str, float]:
    """
    ゾーン別に MAE/RMSE/R² を計算し、ゾーン平均を返す。

    リーク防止: 評価は予測値と実測値の比較のみ。train の統計量を使わない。
    MAPE は使わない（ゼロ需要で発散するため）。

    Parameters
    ----------
    df_eval : pd.DataFrame
        date/zone/demand 列を持つ評価用 DataFrame（val または test）。
    y_pred : np.ndarray
        モデルの予測値。df_eval と同じ順序・長さであること。
    split_name : str
        ログ出力用のラベル（"val" または "test"）。

    Returns
    -------
    dict
        {"mae": float, "rmse": float, "r2": float} — ゾーン平均値。
    """
    df_tmp = df_eval[["zone", "demand"]].copy()
    df_tmp["pred"] = y_pred

    zone_metrics: List[Dict[str, float]] = []
    for zone, grp in df_tmp.groupby("zone"):
        y_true_z = grp["demand"].values
        y_pred_z = grp["pred"].values
        mae_z = mean_absolute_error(y_true_z, y_pred_z)
        rmse_z = math.sqrt(mean_squared_error(y_true_z, y_pred_z))
        r2_z = r2_score(y_true_z, y_pred_z)
        zone_metrics.append({"zone": zone, "mae": mae_z, "rmse": rmse_z, "r2": r2_z})
        logger.debug(
            f"  [{split_name}] zone={zone}: MAE={mae_z:.2f}, RMSE={rmse_z:.2f}, R²={r2_z:.4f}"
        )

    agg = {
        "mae": float(np.mean([m["mae"] for m in zone_metrics])),
        "rmse": float(np.mean([m["rmse"] for m in zone_metrics])),
        "r2": float(np.mean([m["r2"] for m in zone_metrics])),
    }
    logger.info(
        f"[{split_name}] ゾーン平均: MAE={agg['mae']:.2f}, "
        f"RMSE={agg['rmse']:.2f}, R²={agg['r2']:.4f}"
    )
    return agg


# ===========================
# Seasonal-Naive ベースライン
# ===========================

class SeasonalNaive:
    """
    demand_lag_7（7日前の需要）をそのまま予測値とする参照モデル。

    ハイパラ調整なし、scaler 不要。
    demand_lag_7 が存在しない条件では demand_lag_1 にフォールバックする。

    リーク防止: 過去のラグ値のみを使う。ラグはすでに特徴量として付与済みであり、
    未来値を参照しない（features.py で shift(lag) により生成済み）。
    """

    def __init__(self) -> None:
        self.lag_col_: str | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "SeasonalNaive":
        # ラグ列の選択（どちらが利用可能か確認）
        if "demand_lag_7" in X.columns:
            self.lag_col_ = "demand_lag_7"
        elif "demand_lag_1" in X.columns:
            self.lag_col_ = "demand_lag_1"
        else:
            raise ValueError("demand_lag_7 も demand_lag_1 も存在しません")
        logger.info(f"SeasonalNaive: 参照列 = {self.lag_col_}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.lag_col_ is None:
            raise RuntimeError("fit() を先に呼んでください")
        return X[self.lag_col_].values


# ===========================
# 日付ブロック CV（パネルデータ用）
# ===========================

def date_block_splits(
    date_values: pd.Series, n_splits: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    日付ブロック単位の TimeSeriesSplit を行い、(train_idx, val_idx) のリストを返す。

    リーク防止: パネルデータ（同一 date に複数ゾーン）に対して行単位の
    TimeSeriesSplit を使うと、fold 境界の同一日が fold-train と fold-val に
    跨って分割され、同日の他ゾーンの情報が学習側に漏れる。これを避けるため、
    まず**ユニーク日付**に対して TimeSeriesSplit を適用し、得られた日付集合で
    行をマスクする。これにより fold 境界が必ず日付境界に一致し、各 fold-val は
    fold-train より厳密に未来になる。

    Parameters
    ----------
    date_values : pd.Series
        train（date 昇順ソート済み）の date 列。
    n_splits : int
        分割数（config.tuning.cv_splits）。

    Returns
    -------
    list of (np.ndarray, np.ndarray)
        各 fold の (train 行位置インデックス, val 行位置インデックス)。
        RandomizedSearchCV の cv 引数にそのまま渡せる。
    """
    dates = pd.to_datetime(pd.Series(date_values).reset_index(drop=True)).to_numpy()
    unique_days = np.unique(dates)  # ソート済みのユニーク日付
    tscv = TimeSeriesSplit(n_splits=n_splits)

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for tr_day_idx, va_day_idx in tscv.split(unique_days):
        tr_mask = np.isin(dates, unique_days[tr_day_idx])
        va_mask = np.isin(dates, unique_days[va_day_idx])
        splits.append((np.where(tr_mask)[0], np.where(va_mask)[0]))
    return splits


# ===========================
# 1条件のフル学習・評価
# ===========================

def run_condition(
    condition: str,
    cfg: dict,
) -> List[Dict]:
    """
    1つのアブレーション条件（A/B/C/D/E）について3モデルを学習・評価し、
    結果行のリストを返す。

    手順:
      1. load_splits で日付分割。
      2. SeasonalNaive: train で fit、val/test で predict。
      3. Ridge: Pipeline[StandardScaler→Ridge]。train 内 TimeSeriesSplit で CV。
                scaler は Pipeline 内で train のみに fit される（val/test は transform のみ）。
      4. XGBoost: scaler なし。train 内 TimeSeriesSplit で CV。
      5. 各モデルで val/test のゾーン別評価→集約。

    リーク防止:
      - scaler.fit は Pipeline 内で train の CV fold のみに閉じる。
      - TimeSeriesSplit は train を date 昇順でさらに分割するため、
        CV fold 内でも未来が train に混入しない。
      - val/test は評価のみに使用（CV に混入しない）。
    """
    n_iter = cfg["tuning"]["n_iter"]
    cv_splits = cfg["tuning"]["cv_splits"]
    random_state = cfg["tuning"]["random_state"]

    logger.info(f"=== 条件 {condition} の学習開始 ===")

    # --- 1. 分割（日付ベース、ランダム分割なし）---
    train, val, test = load_splits(condition, cfg)
    logger.info(
        f"  train={len(train):,}, val={len(val):,}, test={len(test):,}"
    )

    # --- 日付ブロック CV の設定 ---
    # train を date 昇順でソート（時間順ブロック分割の前提）。
    train_sorted = train.sort_values("date").reset_index(drop=True)

    # リーク防止: パネルデータ（同一 date に 10 ゾーン）に行単位の TimeSeriesSplit
    # を使うと、fold 境界の同一日が fold-train と fold-val に跨って分割され、同日の
    # 他ゾーン情報が学習側に漏れる。これを防ぐため、ユニーク日付に対して
    # TimeSeriesSplit を適用し、fold 境界を必ず日付境界に一致させる。
    # 得られた (train_idx, val_idx) を両モデルの RandomizedSearchCV に共通で渡す。
    cv_date_blocks = date_block_splits(train_sorted["date"], cv_splits)

    # --- 名義変数 zone のダミー変数化（Ridge / XGBoost 共通） ---
    # zone 以外の特徴量列。zone はダミー化して別途結合するためここから除く。
    model_feat_cols = [
        c for c in train.columns
        if c not in {"date", "zone", "demand"}
    ]
    # ダミーのカテゴリは train から固定（val/test を覗かず、全 split で同一列構成）。
    zone_categories = sorted(train["zone"].unique().tolist())

    results = []

    # ===========================
    # モデル 1: Seasonal-Naive
    # ===========================
    logger.info(f"  [{condition}] SeasonalNaive 学習中...")

    # SeasonalNaive は lag 列を DataFrame から直接参照するため、
    # 全特徴量列（lag 列含む） + zone を X として渡す
    sn_feat_cols_base = [c for c in train.columns if c not in {"date", "demand"}]
    # zone も含める（predict で lag 列にアクセスするため全列が必要）
    sn_feat_cols = sn_feat_cols_base

    X_train_sn = train[sn_feat_cols]
    X_val_sn = val[sn_feat_cols]
    X_test_sn = test[sn_feat_cols]

    sn = SeasonalNaive()
    sn.fit(X_train_sn)

    val_pred_sn = sn.predict(X_val_sn)
    test_pred_sn = sn.predict(X_test_sn)

    val_metrics_sn = evaluate_by_zone(val, val_pred_sn, f"{condition}/SeasonalNaive/val")
    test_metrics_sn = evaluate_by_zone(test, test_pred_sn, f"{condition}/SeasonalNaive/test")

    results.append({
        "condition": condition,
        "model": "SeasonalNaive",
        "val_mae": val_metrics_sn["mae"],
        "val_rmse": val_metrics_sn["rmse"],
        "val_r2": val_metrics_sn["r2"],
        "test_mae": test_metrics_sn["mae"],
        "test_rmse": test_metrics_sn["rmse"],
        "test_r2": test_metrics_sn["r2"],
    })

    # ===========================
    # モデル 2: Ridge（線形系、Pipeline で scaler 内包）
    # ===========================
    logger.info(f"  [{condition}] Ridge 学習中 (n_iter={n_iter}, cv_splits={cv_splits})...")

    # 名義変数 zone はダミー変数化して投入する（drop_first=True で k-1 個）。
    # zone 以外の特徴量列に zone の one-hot ダミーを結合して X を作る。
    X_train_ridge = build_zone_onehot_matrix(train_sorted, model_feat_cols, zone_categories)
    y_train_ridge = train_sorted["demand"]
    X_val_ridge = build_zone_onehot_matrix(val, model_feat_cols, zone_categories)
    X_test_ridge = build_zone_onehot_matrix(test, model_feat_cols, zone_categories)

    # Pipeline[StandardScaler → Ridge]
    # リーク防止: StandardScaler は Pipeline 内部で CV の train fold のみに fit される。
    #            val fold・val/test スプリットには transform のみが適用される。
    ridge_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge()),
    ])

    # ハイパラ探索空間
    ridge_param_dist = {
        "ridge__alpha": loguniform(1e-3, 1e4),  # 0.001 〜 10000 の対数一様分布
    }

    # RandomizedSearchCV: n_iter は全モデルで同一（探索予算の公平比較）
    # CV は train 内 TimeSeriesSplit のみ（val/test を混入しない）
    ridge_search = RandomizedSearchCV(
        estimator=ridge_pipe,
        param_distributions=ridge_param_dist,
        n_iter=n_iter,         # 全モデルで同一の n_iter
        cv=cv_date_blocks,     # 日付ブロック CV（同一日の fold 跨ぎを防止）
        scoring="neg_mean_absolute_error",
        random_state=random_state,
        n_jobs=-1,
        refit=True,            # 最良パラメータで train 全体に再フィット
    )
    ridge_search.fit(X_train_ridge, y_train_ridge)
    logger.info(f"  [{condition}] Ridge 最良パラメータ: {ridge_search.best_params_}")

    val_pred_ridge = ridge_search.predict(X_val_ridge)
    test_pred_ridge = ridge_search.predict(X_test_ridge)

    val_metrics_ridge = evaluate_by_zone(val, val_pred_ridge, f"{condition}/Ridge/val")
    test_metrics_ridge = evaluate_by_zone(test, test_pred_ridge, f"{condition}/Ridge/test")

    results.append({
        "condition": condition,
        "model": "Ridge",
        "val_mae": val_metrics_ridge["mae"],
        "val_rmse": val_metrics_ridge["rmse"],
        "val_r2": val_metrics_ridge["r2"],
        "test_mae": test_metrics_ridge["mae"],
        "test_rmse": test_metrics_ridge["rmse"],
        "test_r2": test_metrics_ridge["r2"],
    })

    # ===========================
    # モデル 3: XGBoost（木系、scaler 不要）
    # ===========================
    logger.info(f"  [{condition}] XGBoost 学習中 (n_iter={n_iter}, cv_splits={cv_splits})...")

    # 木系も名義変数 zone はダミー変数化して投入（全モデルで表現を統一）。
    # 以前は zone を生の整数のまま渡していたが、Ridge と同じ one-hot 表現に揃える。
    X_train_xgb = build_zone_onehot_matrix(train_sorted, model_feat_cols, zone_categories)
    y_train_xgb = train_sorted["demand"]
    X_val_xgb = build_zone_onehot_matrix(val, model_feat_cols, zone_categories)
    X_test_xgb = build_zone_onehot_matrix(test, model_feat_cols, zone_categories)

    # XGBoost: scaler 不要（木系はスケール不変）
    xgb_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )

    # ハイパラ探索空間
    xgb_param_dist = {
        "n_estimators": randint(50, 500),          # 50〜499
        "max_depth": randint(2, 9),                # 2〜8
        "learning_rate": loguniform(0.01, 0.3),    # 0.01〜0.3 対数一様
        "subsample": uniform(0.6, 0.4),            # 0.6〜1.0
        "colsample_bytree": uniform(0.6, 0.4),     # 0.6〜1.0
        "reg_alpha": loguniform(1e-4, 10.0),       # L1正則化
        "reg_lambda": loguniform(1e-4, 10.0),      # L2正則化
    }

    # RandomizedSearchCV: n_iter は全モデルで同一（探索予算の公平比較）
    # CV は train 内 TimeSeriesSplit のみ（val/test を混入しない）
    xgb_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=xgb_param_dist,
        n_iter=n_iter,         # 全モデルで同一の n_iter
        cv=cv_date_blocks,     # 日付ブロック CV（同一日の fold 跨ぎを防止）
        scoring="neg_mean_absolute_error",
        random_state=random_state,
        n_jobs=-1,
        refit=True,            # 最良パラメータで train 全体に再フィット
    )
    xgb_search.fit(X_train_xgb, y_train_xgb)
    logger.info(f"  [{condition}] XGBoost 最良パラメータ: {xgb_search.best_params_}")

    val_pred_xgb = xgb_search.predict(X_val_xgb)
    test_pred_xgb = xgb_search.predict(X_test_xgb)

    val_metrics_xgb = evaluate_by_zone(val, val_pred_xgb, f"{condition}/XGBoost/val")
    test_metrics_xgb = evaluate_by_zone(test, test_pred_xgb, f"{condition}/XGBoost/test")

    results.append({
        "condition": condition,
        "model": "XGBoost",
        "val_mae": val_metrics_xgb["mae"],
        "val_rmse": val_metrics_xgb["rmse"],
        "val_r2": val_metrics_xgb["r2"],
        "test_mae": test_metrics_xgb["mae"],
        "test_rmse": test_metrics_xgb["rmse"],
        "test_r2": test_metrics_xgb["r2"],
    })

    return results


# ===========================
# メインエントリーポイント
# ===========================

def main() -> None:
    """
    5条件 × 3モデルのフル学習・評価を実行し、
    results/ablation_table.csv を生成する。

    条件 E（ゾーン解像イベント）は dataset_E.parquet が存在する場合のみ評価する。

    評価方針:
      - val: ハイパラ選択・モデル選択に使用。
      - test: 最終報告のみ（1度だけ評価）。val によるモデル選択後に評価。
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_config()

    # 条件 E は features.py がゾーン解像イベント列を持つときだけ生成されるため、
    # データセットが存在する条件だけを評価対象にする。
    features_dir = Path(cfg["paths"]["features_dir"])
    if not features_dir.is_absolute():
        features_dir = Path(__file__).resolve().parent.parent / features_dir
    conditions = [
        c for c in ("A", "B", "C", "Cprime", "D", "E")
        if (features_dir / f"dataset_{c}.parquet").exists()
    ]

    all_results = []
    for condition in conditions:
        cond_results = run_condition(condition, cfg)
        all_results.extend(cond_results)

    # --- DataFrame に変換 ---
    df_result = pd.DataFrame(all_results)

    # --- 列の順序を整理 ---
    col_order = [
        "condition", "model",
        "val_mae", "val_rmse", "val_r2",
        "test_mae", "test_rmse", "test_r2",
    ]
    df_result = df_result[col_order]

    # --- 数値を丸める（視認性のため小数点3桁）---
    for col in ["val_mae", "val_rmse", "val_r2", "test_mae", "test_rmse", "test_r2"]:
        df_result[col] = df_result[col].round(3)

    # --- 保存 ---
    results_path = Path(cfg["paths"]["results"])
    if not results_path.is_absolute():
        project_root = Path(__file__).resolve().parent.parent
        results_path = project_root / results_path

    results_path.parent.mkdir(parents=True, exist_ok=True)
    df_result.to_csv(results_path, index=False, encoding="utf-8")

    logger.info(f"ablation_table.csv を保存しました: {results_path}")

    # --- コンソール出力 ---
    print("\n===== Ablation Table =====")
    print(df_result.to_string(index=False))
    print(f"\n保存先: {results_path}")


if __name__ == "__main__":
    main()
