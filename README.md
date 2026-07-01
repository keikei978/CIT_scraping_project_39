# 天候 × 移動需要 — 研究リポジトリ

天候などの外部情報が移動需要に効くのか、効くならなぜ「効く／効かない」が分かれるのかを、
3つのサブプロジェクトを通して調べる研究です。

**まず [研究メモ_班員向け.md](研究メモ_班員向け.md) を読んでください**（研究全体の狙いと流れ）。

## 構成

| ディレクトリ | 役割 | 状態 |
|---|---|---|
| [`nyc-taxi-demand/`](nyc-taxi-demand/) | ①タクシー需要 — 外部情報は効かなかった例 | 実装・分析まで完了 |
| [`nyc-citibike-demand/`](nyc-citibike-demand/) | ②シェアサイクル需要 — 天候が効いた例 | 実装計画確定 |
| [`nyc-weather-discretion/`](nyc-weather-discretion/) | ③「やめられる移動か」検証 — 差の正体を探す | 最新・Phase A |

①で疑問が出て、②で手がかりが出て、③でその要因を突き止める、という一本の流れになっています。

## データ源（すべて公開・APIキー不要）

- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)（全プロジェクトの天候）
- [NYC TLC Yellow Taxi Trip Records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)（①）
- [NYC Open Data — Permitted Event Information](https://data.cityofnewyork.us/City-Government/NYC-Permitted-Event-Information-Historical/bkfu-528j)（①）
- [Citi Bike System Data](https://citibikenyc.com/system-data)（②③）

## 備考

- `data/`・`.venv/`・生成物（parquet 等）は `.gitignore` で除外しています（再生成可能・サイズ大）。
- 各プロジェクトは共通の規律（データリーク検査 / 学習期間のみで前処理を決める / 時系列分割）を守っています。
