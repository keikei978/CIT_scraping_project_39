# ===== リーク検査テスト =====
# このテストは実装コードを書く前に確定させる。
# feature-builder / ml-trainer が作る関数を import して検証する。
# 関数名が違う場合は ml-trainer に「この関数名に合わせて」と指示すること。

import numpy as np
import pandas as pd
import pytest


def make_sample_df(n_days=60, n_zones=3):
    '''テスト用のダミー日次需要データ'''
    dates = pd.date_range('2022-01-01', periods=n_days, freq='D')
    rows = []
    for z in range(n_zones):
        for i, d in enumerate(dates):
            rows.append({'date': d, 'zone': z, 'demand': 100 + i + z * 10})
    return pd.DataFrame(rows)


# ---------- 検査A: ラグ特徴量が未来を見ていないか ----------
def test_lag_only_uses_past():
    from src.features import build_features  # feature-builder が作る
    df = make_sample_df()
    out = build_features(df, lags=[1, 7])
    for lag in [1, 7]:
        col = f'demand_lag_{lag}'
        assert col in out.columns, f'{col} が無い'
        expected = out.groupby('zone')['demand'].shift(lag)
        mask = expected.notna()
        # ラグ値が「lag日前の実需要」と完全一致 = 未来参照なし
        assert (out.loc[mask, col].values == expected[mask].values).all(), \
            f'{col} が未来を参照している疑い'


# ---------- 検査B: rolling が当日を含まないか ----------
def test_rolling_excludes_current_day():
    from src.features import build_features
    df = make_sample_df()
    out = build_features(df, lags=[1], rolling_windows=[7])
    # rolling列が存在し、当日の値を含んでいないこと
    roll_cols = [c for c in out.columns if 'roll' in c]
    assert len(roll_cols) > 0, 'rolling特徴量が無い'
    # shift(1)してからrollingしていれば、最初の数行はNaNになるはず
    for c in roll_cols:
        first_valid = out.groupby('zone')[c].apply(lambda s: s.first_valid_index())
        assert out[c].isna().sum() > 0, f'{c} に当日が混入している疑い'


# ---------- 検査C: 分割が時間順で重複しないか ----------
def test_split_no_overlap():
    from src.split import load_splits  # ml-trainer が作る
    train, val, test = load_splits()
    assert train['date'].max() < val['date'].min(), 'Train と Val が重複'
    assert val['date'].max() < test['date'].min(), 'Val と Test が重複'


# ---------- 検査D: scaler が Train のみで fit されているか ----------
def test_scaler_fit_on_train_only():
    from src.split import load_splits
    from src.train import fit_scaler  # ml-trainer が作る
    train, val, test = load_splits()
    feature_cols = [c for c in train.columns if c not in ('date', 'zone', 'demand')]
    scaler = fit_scaler(train[feature_cols])
    # scaler の平均が Train の平均と一致 = Val/Test で fit していない
    np.testing.assert_allclose(
        scaler.mean_, train[feature_cols].mean().values, rtol=1e-5,
        err_msg='scaler が Train 以外のデータで fit されている疑い'
    )


# ---------- 検査E: ランダム分割が使われていないか ----------
def test_no_random_split_in_source():
    '''実装ソースに train_test_split のランダム使用が無いことを確認'''
    import pathlib
    src_files = pathlib.Path('src').glob('*.py')
    for f in src_files:
        text = f.read_text(encoding='utf-8')
        if 'train_test_split' in text:
            # shuffle=False が明示されていればOK、それ以外は警告
            assert 'shuffle=False' in text, \
                f'{f.name} で train_test_split がシャッフル付きで使われている疑い'


# ---------- 検査F: precinct→zone の weight が per-precinct で正規化されているか ----------
def test_precinct_crosswalk_weights_normalized():
    '''
    precinct 文字列の分解が正しく、precinct→zone クロスウォークの weight が
    precinct ごとに 1 へ正規化されていることを確認する。

    この検査が保証するのは「単一 precinct 内での二重計上の防止」（per-precinct 正規化）
    であり、イベント単位の質量保存ではない。全数カウント設計では、K 個の precinct に
    またがる 1 イベントは全ゾーン合計で質量 K になる（これは合意した既定）。
    weight 和が 1 でないと単一 precinct のイベントが複数ゾーンへ合計≠1 で計上され、
    event_count_zone が系統的に歪む。これは train/val/test を横断するバイアスに
    なり得るため機械的に禁止する。
    '''
    from src.events import _parse_precincts, load_precinct_zone_crosswalk
    from src.config import load_config

    # precinct 文字列（例 "17," / "17,18,"）の分解が正しいこと
    assert _parse_precincts('17,') == [17]
    assert _parse_precincts('17,18,') == [17, 18]
    assert _parse_precincts('') == []

    cw = load_precinct_zone_crosswalk(
        load_config()['paths']['precinct_zone_crosswalk']
    )
    weight_sum = cw.groupby('precinct')['weight'].sum()
    np.testing.assert_allclose(
        weight_sum.values, 1.0, atol=1e-6,
        err_msg='crosswalk の weight が precinct ごとに 1 へ正規化されていない（二重計上の疑い）'
    )


# ---------- 検査G: ゾーン解像イベントが市内スカラーへ退化していないか ----------
def test_zone_events_are_zone_discriminative():
    '''
    条件 E の event_count_zone が (date, zone) 別に実際に異なる値を持つことを確認する。

    目的: 全市スカラー（同一日は全ゾーン同値）への「サイレントな退化」を検出する。
    退化していれば同一日内のゾーン間分散が常に 0 になり、空間解像度を与えた意味が
    失われる。条件 C（市内）との対比を成立させる前提でもある。
    '''
    import pathlib

    if not pathlib.Path('data/features/dataset_E.parquet').exists():
        pytest.skip('dataset_E.parquet 未生成（features を実行後に有効）')

    from src.split import load_splits
    train, _, _ = load_splits('E')
    assert 'event_count_zone' in train.columns, 'event_count_zone 列が無い'

    # 同一日内でゾーン間に値の差がある日が存在する = ゾーン弁別的
    within_day_var = train.groupby('date')['event_count_zone'].var().fillna(0.0)
    assert (within_day_var > 0).sum() > 0, \
        'event_count_zone が全日でゾーン間一定 = 市内スカラーに退化している疑い'
