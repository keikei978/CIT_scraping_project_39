# NYC Taxi Demand Forecasting

ニューヨーク市マンハッタンの主要タクシーゾーンについて、**日次の乗車需要を予測**する時系列回帰プロジェクト。
中心テーマは **「外部情報（気象・イベント）を加えると需要予測の精度は上がるのか？」** をアブレーション（A/B/C/C′/D/E の6条件比較）で定量検証することにある。特に「イベントに**空間解像度**を与えれば効くのか」を、許可イベントを taxi zone 別に割り付けた条件 E と統制条件 C′ で直接検証している。

設計上の最重要原則は **データリークの厳格な排除**で、専用のレビュー監査とリーク検査テスト（`tests/test_no_leakage.py`）で機械的に担保している。

---

## 主要な結論（重要）

> **外部情報（気象・許可イベント件数）は、市全体スカラー・Manhattan限定スカラー・ゾーン解像のいずれの形でも、日次×ゾーンの需要予測精度をほぼ改善しなかった。**

XGBoost を基準に見ると、時間特徴のみ（条件 A）の test R² = 0.736 に対し、気象 B・イベント各形（C/C′/E）・全部 D はいずれも同等（0.726〜0.744、誤差範囲）。Ridge では条件間の test R² は 0.620〜0.626（MAE 差 1% 未満）でほぼ平坦。

とりわけ「外部情報に**空間解像度**を与えれば効くのではないか」という仮説を、許可イベントを taxi zone 別に割り付けた**条件 E**で直接検証した。結果、ゾーン解像にしても改善しなかった。交絡を避けるため統制条件 **C′（Manhattan限定の同一イベントデータを日次スカラーに集約）** を置き、**E vs C′ で「空間解像度の効果のみ」を分離**したが、ベースライン（zone+カレンダー+ラグ）への線形の増分 R² は **+0.000**（C′ も E も A と同値）。XGBoost で C′/E の test がごく僅かに A を上回る（0.744/0.740）が、Ridge では平坦・線形増分ゼロ・val/test が条件間で不整合のため、B/C/D と同じ**誤差帯のノイズ**と判断する。

> 注: 名義変数 zone（LocationID）は全モデルで one-hot（ダミー変数, drop_first=True）化して投入している。以前は Ridge で zone を除外・XGBoost で生の整数を使っていたが、表現を統一して再学習した結果が下表。

### なぜ効かなかったか（2層の限界）

**第1層: 市全体スカラーは空間弁別力を持たない。**
気象（Open-Meteo マンハッタン中心1点）も、当初のイベント件数（NYC Open Data を市全体で日次集計）も、**「日付ごとに1値」で同一日は全ゾーン同一値**。一方ターゲット需要は**ゾーン差が支配的**（同一日でもゾーン間で数倍違う）。ゾーン弁別力のないスカラーは、モデルが既に持つ時間特徴（ラグ・曜日・季節）以上をほとんど加えられない。

**第2層: 空間解像度を与えても、許可イベントというデータ種別が需要ホットスポットと一致しない。**
そこで許可イベントを `police_precinct` → taxi zone の面積重みクロスウォークでゾーン別に割り付け（**条件 E**）、空間解像度を実際に与えた（`event_count_zone` の同一日内ゾーン間分散は 0 → 640.7 に上昇＝本物のゾーン弁別性）。しかし精度は改善しなかった。原因は、**許可イベントの実体が公園・レクリエーション許可中心**で、その質量が Central Park（16.8%）・Randalls Island（11.7%）・East Harlem North（10.6%）等の公園系ゾーンに集中し、**モデル対象の高需要商業ゾーン（Midtown・Times Sq・UES 等）に落ちる質量は全体の 5.5% に過ぎない**ためである。空間解像度は本物になったが、乗せる信号そのものが対象ゾーンに乏しい。

つまり限界は「空間粒度」だけでなく「**外部データの種別と対象ゾーンの空間的不一致**」にある。改善には、需要ホットスポットに実在するイベント＝**会場別イベント（MSG・リンカーンセンター・劇場等）** を会場座標でゾーンに割り付けることが要る（今後の課題）。実際、対象10ゾーンの中でイベント質量がやや乗ったのは Penn Station/MSG・Lincoln Square East・Times Sq といった会場性ゾーンだった。

---

## アブレーション結果（`results/ablation_table.csv`）

評価はゾーン別に算出してからゾーン平均（MAPE はゼロ需要で発散するため不使用）。
val=2024 上半期でハイパラ選択、test=2024 下半期は最終評価のみ。

6条件: A 時間のみ / B +気象 / C +イベント(全市スカラー) / **C′ +イベント(Manhattanスカラー)** / D +気象+イベント / **E +イベント(ゾーン解像)**。
C/C′/E はいずれも許可イベントだが、C=全5区の整数件数、C′=C と同じ Manhattan イベントを日次スカラーに集約、E=同データを taxi zone 別に面積按分。**E vs C′ が空間解像度の純粋比較**、C′ vs C が区スコープの比較。

| 条件 | モデル | val MAE | val RMSE | val R² | test MAE | test RMSE | test R² |
|---|---|---|---|---|---|---|---|
| A 時間のみ | SeasonalNaive | 429.7 | 579.5 | 0.454 | 598.6 | 912.5 | 0.103 |
| A 時間のみ | Ridge | 381.3 | 489.6 | 0.549 | 425.0 | 556.1 | 0.620 |
| A 時間のみ | **XGBoost** | **307.4** | **408.8** | **0.704** | **346.1** | **461.2** | **0.736** |
| B +気象 | XGBoost | 299.5 | 396.4 | 0.724 | 355.2 | 472.8 | 0.726 |
| C +イベント(全市) | XGBoost | 313.1 | 412.0 | 0.697 | 349.2 | 466.8 | 0.737 |
| C′ +イベント(Manhattanスカラー) | XGBoost | 303.6 | 398.4 | 0.720 | 341.1 | 458.0 | 0.744 |
| D +気象+イベント | XGBoost | 303.7 | 399.9 | 0.724 | 354.3 | 473.1 | 0.729 |
| E +イベント(ゾーン解像) | XGBoost | 309.8 | 408.9 | 0.703 | 342.5 | 456.2 | 0.740 |

（Ridge は E/C′ とも A とほぼ同値: test R² A 0.620 / C′ 0.621 / E 0.620。全行は CSV 参照。SeasonalNaive は demand_lag_7 のみ参照のため全6条件で同一値。）

要点:
- モデル性能順は **XGBoost > Ridge > SeasonalNaive**（時間特徴の非線形な効きが大きい）。
- **外部情報の追加（A→B/C/C′/D/E）は精度をほぼ動かさない**（上記「主要な結論」参照）。
- **E vs C′（空間解像度の純粋比較）でも改善なし**: ゾーン解像にしても許可イベントは効かない（第2層の限界＝公園系集中による空間的不一致）。

---

## プロジェクト構造

```
nyc-taxi-demand/
├── config/
│   ├── config.yaml             # 全パラメータの単一の真実（期間・ゾーン・分割・特徴量・チューニング・クロスウォークパス）
│   └── precinct_zone_crosswalk.csv  # precinct→LocationID の面積重み（条件C′/E用・固定成果物）
├── scripts/
│   └── build_precinct_zone_crosswalk.py  # [一度だけ] NYPD precinct × taxi zone の面積重ね合わせで上記CSVを生成（geopandasはここ限定）
├── src/
│   ├── config.py               # config.yaml ローダ
│   ├── ingest.py    [Phase 1]  # TLC yellow taxi 取得→日次×ゾーン集計→train期間でゾーン自動選択→完全グリッド
│   ├── weather.py   [Phase 2]  # Open-Meteo archive-api から日次気象
│   ├── events.py    [Phase 2]  # NYC Open Data から ①全市スカラー件数 fetch_events（C/D用）
│   │                           #                  ②ゾーン解像 fetch_events_by_zone（precinct→zone面積按分, E用）
│   ├── merge.py     [Phase 2]  # 需要+気象+イベントを結合（event_countはdate, event_count_zoneは(date,zone)）
│   │                           #   + event_count_manhattan（C′用の日次合算スカラー）→ merged.parquet
│   ├── features.py  [Phase 3]  # ラグ/rolling/周期(sin,cos)/祝日 + 6条件データセット
│   ├── split.py     [Phase 4]  # 日付による厳密な train/val/test 分割
│   └── train.py     [Phase 5-6]# 学習・RandomizedSearchCV・評価・ablation 出力（zone は全モデルで one-hot 化）
├── tests/test_no_leakage.py    # リーク検査7種（A〜E＋クロスウォーク正規化・ゾーン弁別性）
├── data/                       # 中間生成物（.gitignore 対象）
│   ├── daily_demand.parquet
│   ├── merged.parquet          # 列に event_count / event_count_zone / event_count_manhattan を含む
│   ├── _manhattan_daily_all.parquet  # 全67ゾーン中間（ゾーン再選択用、再DL回避）
│   ├── geo/                    # クロスウォーク生成用のシェープファイル（TLC taxi zones / NYPD precincts）
│   └── features/dataset_{A,B,C,Cprime,D,E}.parquet
└── results/ablation_table.csv  # 最終成果物
```

### 設定（config/config.yaml）
- 期間: 2022-01-01 〜 2024-12-31
- ゾーン: Manhattan の需要上位 10（exclude_ids=[132,138]、**train 期間のみで選定**）
- 分割: train ≤ 2023-12-31 / val ≤ 2024-06-30 / test = 2024 下半期
- 特徴量: ラグ [1,2,3,7,14]、rolling [7]
- チューニング: n_iter=20、cv_splits=5、random_state=42（全モデル統一＝公平比較）

---

## データリーク防止の設計

本プロジェクトの中核。各フェーズで以下を担保し、`tests/test_no_leakage.py` の5検査で機械的に検証している。

| 防止策 | 実装箇所 | 検査 |
|---|---|---|
| ラグは正の shift のみ（未来参照しない） | features.py | A |
| rolling は shift(1)＋ゾーン内 transform（当日除外・ゾーン境界を跨がない） | features.py | B |
| 分割は日付で厳密に（ランダム分割禁止） | split.py | C / E |
| StandardScaler は train のみで fit（Pipeline 内で CV fold ごとに再fit） | train.py | D |
| CV は日付ブロック TimeSeriesSplit（同一日が fold-train/val に跨らない） | train.py | — |
| ゾーン選定は train 期間の需要のみで実施（test を覗かない） | ingest.py | — |
| イベント取得失敗月はサイレント 0 埋めせず fail（系統的バイアス防止） | events.py | — |
| precinct→zone クロスウォークは weight を precinct ごとに 1 へ正規化（単一 precinct 内の二重計上防止） | build_precinct_zone_crosswalk.py | F |
| (date,zone) 結合前に zone を int 明示キャスト（dtype 不一致で E が A へ退化するのを防止） | merge.py | G |
| ゾーン解像イベントは同日値の外生情報（許可は事前申請済み・demand を一切参照しない）→ 未来参照リークに非該当 | events.py / features.py | — |

---

## 実行方法

```bash
# 依存は .venv に導入済み（pandas, numpy, scikit-learn, xgboost, holidays, pyarrow, requests, PyYAML, pytest）
# クロスウォーク生成のみ geopandas/shapely/pyproj/pyogrio を使う（本体パイプラインには不要）

# 一度だけ: precinct→zone クロスウォークを生成（条件 C′/E の前提。geopandas 必要）
#   data/geo/ の TLC taxi zones と NYPD precincts シェープファイルから面積重みを算出
.venv/Scripts/python.exe scripts/build_precinct_zone_crosswalk.py  # config/precinct_zone_crosswalk.csv

# Phase 1-2: データ取得・結合（ネットワーク必要。TLC 全36ヶ月の DL を含む）
.venv/Scripts/python.exe -m src.ingest      # data/daily_demand.parquet
.venv/Scripts/python.exe -m src.merge       # data/merged.parquet（event_count_zone/_manhattan 含む）

# Phase 3: 特徴量・6条件データセット
.venv/Scripts/python.exe -m src.features    # data/features/dataset_{A,B,C,Cprime,D,E}.parquet

# Phase 4-6: 学習・評価
.venv/Scripts/python.exe -m src.train       # results/ablation_table.csv

# リーク検査
.venv/Scripts/python.exe -m pytest tests/test_no_leakage.py -v
```

---

## 既知の限界・今後の課題

- **外部データの種別と対象ゾーンの空間的不一致**（最重要、上記「なぜ効かなかったか」参照）: ゾーン解像（条件 E）を与えても許可イベントは効かなかった。許可イベントが公園・レク中心で、需要ホットスポット（Midtown 等）に質量が乗らない（対象10ゾーンに全体の 5.5%）ためである。**最有望の次手は会場別イベント（MSG・リンカーンセンター・劇場等）を会場座標でゾーンに割り付けること**。ゾーン×天候の交互作用も候補。
- **イベントのゾーン割付方法**: `event_location` は自由文の会場名で緯度経度を持たないため、`police_precinct` 経由でのみゾーンへ落とせる。precinct→zone は面積按分（precinct 内で需要影響が面積一様と仮定）で、複数 precinct イベントは各 precinct で全数カウントする近似。
- **イベント母集団の差**: 条件 E/C′ は `event_borough='Manhattan'` かつ precinct を持つイベントのみ（inner 結合。Manhattan の precinct 充足率は実測 100%）。全5区を数える条件 C とは母集団が異なるため、空間解像度を公平に切り分ける統制条件として C′（Manhattan限定スカラー）を併置した。クロスウォークに無い precinct は静かに脱落し得る。
- **探索予算の非対称性**: n_iter=20 を全モデルで統一しているが、Ridge（1次元）に対し XGBoost（7次元）は探索空間が広く、同一 n_iter では XGBoost 側の探索密度が低い。「同一予算下での比較」としては公平だが、各モデルのベスト性能比較を狙う場合は次元に応じた予算調整が必要。
- **イベントの日付基準**: 複数日にまたがるイベントを開始日のみでカウントしている（開催中の各日には数えていない）。条件 C も C′ も E も同基準のため、これらの対比は対称。
