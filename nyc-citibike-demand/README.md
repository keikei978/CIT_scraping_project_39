# NYC Citi Bike 需要 × 天候 — ベースライン除去で天候効果を顕在化する

ニューヨークの**シェアサイクル（Citi Bike）の日次乗車需要**を題材に、
**「過去需要ラグが予測を支配して天候の寄与が埋もれる」問題に対し、暦ベースラインを除去し
天候を“アノマリ（平年偏差）”として与えると、天候の寄与が季節交絡なく顕在化するか**を検証した
時系列回帰プロジェクト。

姉妹プロジェクト [nyc-taxi-demand] の**負の結果**（タクシーでは天候・イベントが効かず、
前日需要ラグが標準化係数で支配）との**対比**に位置づく。設計・リーク排除の規律はそれを継承。

---

## 主要な結論（重要）

> **天候感応性の高いシェアサイクルでは、天候が需要予測の主役として顕在化した。**
> 暦のみ（条件A）の test R² ≈ 0.2 に対し、**天候アノマリを足す（条件B）と 0.71〜0.84 へ跳ね上がる**。
> しかも **天候(B) > ラグ(C)**。タクシー（天候≈0・ラグ支配）とは正反対。

### なぜこれが「本物の天候効果」と言えるか（季節交絡の排除）

天候は季節と強く相関する（夏＝暑い＝よく乗る）。生の天候を使うと「天候が季節を代理」して
効果が濁る。本研究は2点でこれを構造的に排した:

1. **天候はアノマリ化**（実測 − day-of-year 平年値、平年は train のみで推定）。季節と直交。
2. **ベースラインに季節（調和3次）＋曜日＋成長トレンドを入れる**（生weatherは入れない）。

その上で、**暦ベースラインを引いた残差**と天候アノマリの相関を見ると（total）:

| 天候アノマリ | 残差との相関 |
|---|---|
| **降水時間** | **−0.65** |
| 降水量 | −0.52 |
| **最高気温** | **+0.45** |
| 風速 | −0.26 |

→ **季節循環を持たない「降水」が最強の相関（−0.65）**。これは天候効果が季節の代理ではなく
**本物**であることの最もクリーンな証拠。

---

## アブレーション結果（`results/ablation_table.csv`）

評価は需要スケールで MAE/RMSE/R²。train=2022-01〜2023-06、val=2023 Q3、test=2023 Q4。
条件 **A=暦のみ / B=+天候アノマリ / C=+ラグ / D=+両方**。BASE=暦ベースラインのみの床。

### test R²（XGBoost）

| 条件 | total | member | casual |
|---|---|---|---|
| BASE 暦床 | 0.53 | 0.56 | 0.40 |
| A 暦のみ | 0.21 | 0.22 | 0.19 |
| **B +天候** | **0.71** | **0.84** | **0.80** |
| C +ラグ | 0.44 | 0.49 | 0.30 |
| D 全部 | 0.74 | 0.69 | 0.73 |

（Linear も同傾向で安定: total A 0.54 → B 0.75 → D 0.76。全行・全モデルは CSV 参照。）

要点:
- **A→B（天候追加）の跳ね上がりが支配的**。天候は需要予測の主役。
- **天候(B) > ラグ(C)**。タクシーと逆（タクシーはラグ ≫ 天候）。
- member/casual 両方で成立。casual の標準化係数では `anom_降水時間`・`anom_最高気温` が
  `lag_y_1` を上回る（天候 > ラグ）。

### 天候ショック日評価（`results/badweather_eval.csv`）

test を「前日から天候が急変した日（降水アノマリの前日差・上位10%）」に絞り、
C（ラグのみ）vs D（ラグ+天候）の MAE を比較:

| target | C: ショック日 MAE | D: ショック日 MAE | 改善 |
|---|---|---|---|
| total | 25,118 | **14,895** | **−41%** |
| member | 19,646 | 11,361 | −42% |
| casual | 6,418 | 4,890 | −24% |

→ **ラグが外す「天候急変日」でこそ天候が効く**。天候はクラスタし lag-1 が平常の天候を
織り込むため、急変日で天候の増分が最大になる。

---

## プロジェクト構造

```
nyc-citibike-demand/
├── config/config.yaml          # 期間・データ源・ベースライン・特徴・分割・評価の単一の真実
├── PLAN.md                     # 実装計画（§9 に原案からの精査反映を明記）
├── src/
│   ├── ingest_citibike.py [P1] # 公式S3年次zip(2022/2023)取得→日次×全合算(member/casual/total)
│   │                           #   レジューム+リトライ+ファイル別集計キャッシュ
│   ├── ingest_weather.py  [P2] # Open-Meteo 過去気象（NYC重心1点）
│   ├── merge.py           [P2] # 需要×天候を date 結合
│   ├── baseline.py        [P3] # ★核: 曜日+平滑季節(調和)+成長トレンド、気象アノマリ（train のみ fit）
│   ├── features.py        [P3] # cal_*/anom_*/lag_* 生成（条件A/B/C/Dは接頭辞選択）
│   ├── split.py           [P4] # 日付による厳密分割
│   ├── train.py           [P4] # 条件×3モデル(Linear/RF/XGB)、需要スケールで評価
│   └── analyze.py         [P5] # 標準化係数・残差×アノマリ相関・天候ショック日評価
├── tests/test_no_leakage.py    # リーク検査6種
├── data/                       # 中間生成物（.gitignore 対象）
└── results/                    # ablation_table / standardized_coef / residual_anomaly / badweather_eval
```

### 設定の要点
- 期間: 2022-01-01〜2023-12-31（730日, 平均 88,965 rides/日, member率 0.80）
- 目的変数: 需要 y（total / member / casual）に**一本化**（残差化は cal_* 特徴として内包）
- 分割: train ≤ 2023-06-30 / val ≤ 2023-09-30 / test = 2023 Q4
- 天候: アノマリ（平年偏差, train のみで climatology を fit）

---

## データリーク・交絡防止（`tests/test_no_leakage.py` 6検査）

| 防止策 | 検査 |
|---|---|
| ラグは正の shift のみ（未来非参照） | A |
| rolling は shift(1) 後（当日除外） | B |
| 気象アノマリの climatology は train のみで fit | C |
| ベースラインは train のみで fit（val/test は適用のみ） | D |
| **暦特徴に生 weather を混入させない（交絡防止）。weather は anom_* のみ** | E |
| ランダム分割禁止（日付で厳密に） | F |
| 取得失敗・欠損日はサイレント 0 埋めせず RuntimeError | — |

---

## 実行方法

```bash
# 依存（pandas, numpy, scikit-learn, xgboost, holidays, pyarrow, requests, PyYAML, pytest）
# ※現状は姉妹プロジェクト ../nyc-taxi-demand/.venv を流用

# P1-2 取得（ネットワーク必要。Citi Bike 年次zip 計2.8GB＝レジューム対応）
python -m src.ingest_citibike   # data/daily_demand.parquet
python -m src.ingest_weather    # data/weather.parquet
python -m src.merge             # data/merged.parquet

# P3 特徴量（暦/天候アノマリ/ラグ）
python -m src.features          # data/features/dataset_{total,member,casual}.parquet

# P4-5 学習・評価・分析
python -m src.train             # results/ablation_table.csv
python -m src.analyze           # standardized_coef / residual_anomaly / badweather_eval

# リーク検査
python -m pytest tests/test_no_leakage.py -v
```

---

## 既知の限界・今後の課題

- **空間は扱わない**（全システム合算）。前プロジェクトの空間ミスマッチの轍を避けたため。
  駅レベル×会場近接などの空間方向は将来拡張（リークと解像度に注意して）。
- **暦ベースラインの季節は調和3次**。条件A（暦のみ）が test で弱めなのは、3次では捉えきれない
  季節形状もあるため。ただし最強の天候相関が**非季節の降水**である点から、天候効果の本質は揺るがない。
- **member/casual の対比**は標準化係数・天候ショック日で確認したが、用途別のさらに踏み込んだ
  分解（電動アシスト比率など）は今後の課題。
- **探索予算**は全モデルで n_iter 統一（同一予算下の公平比較）。小データ（train=532日）のため
  RF/XGB は分散が大きく、解釈は線形係数を主軸に読む。
