"""PLAN §10.4 stage 2 (ridge only; gbm failed the stage-1 rule): train the M2 residual ridge on
origins at EVERY hour instead of 23:00 only. Evaluation origins stay the 23:00 sets A (49) and B (9).
(The plan named two files, forecast_hourly.py + exp_hourly.py; merged into this one for simplicity.)

Row i = an origin at hour index i. Features and the M1 baseline at i use observations <= i only, so they
are computed once on the full train and sliced; training rows for validation origin c need i + 72 <= c.
Output: logs/hourly_results.csv. Existing files are not modified.
"""
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import forecast as F
from starter import load_data

H, K, ALPHA_DECAY = 72, 5, 0.97          # M1_BEST: k=5, m=1, alpha=0.97
FIRST = 167                               # 168h of history needed (same as the 23:00 version's first usable day)
RIDGE_ALPHAS = [1, 10, 100]


def origins(a, b):
    return list(pd.date_range(f'{a} 23:00', f'{b} 23:00', freq='3D'))


A1, A2, B = origins('2013-11-01', '2014-02-25'), origins('2014-11-01', '2014-11-28'), origins('2014-12-01', '2014-12-25')
A = A1 + A2


def hourly_tables(df):
    """Per hour index i: M1 forecast (N, 72), 39 features (N, 39), targets (N, 72). Rows < FIRST are NaN."""
    t, td, p = (df[c].to_numpy(float) for c in ('temperature_c', 'dew_point_c', 'pressure_hpa'))
    N = len(t)
    idx = np.arange(N)
    # B5[n] = mean of t[n], t[n-24], ..., t[n-96] = 5-day mean at n's clock hour (window ending at n)
    B5 = np.full(N, np.nan)
    B5[96:] = np.mean([t[96 - 24 * j: N - 24 * j] for j in range(K)], axis=0)
    h = np.arange(1, H + 1)
    ok = idx >= FIRST
    i = idx[ok]
    # profile_i[clock of i+h] = B5[i - ((-h) mod 24)];  anomaly = t[i] - B5[i]
    base = np.full((N, H), np.nan)
    base[ok] = B5[i[:, None] - ((-h) % 24)[None, :]] + (t[i] - B5[i])[:, None] * ALPHA_DECAY ** h
    # last 24h anomalies ordered oldest -> newest: t[i-L] - B5[i-L], L = 23..0
    L = np.arange(23, -1, -1)
    anom24 = t[i[:, None] - L] - B5[i[:, None] - L]
    c = np.concatenate([[0], np.cumsum(t)])
    cd = np.concatenate([[0], np.cumsum(t - td)])
    mean_last = lambda cs, n: (cs[i + 1] - cs[i + 1 - n]) / n
    win = t[i[:, None] - np.arange(24)]
    doy = 2 * np.pi * df.timestamp.dt.dayofyear.to_numpy()[i] / 365.25
    hr = 2 * np.pi * df.timestamp.dt.hour.to_numpy()[i] / 24
    feats = np.full((N, 39), np.nan)
    feats[ok] = np.column_stack([
        anom24, mean_last(c, 24), mean_last(c, 72), mean_last(c, 168), t[i] - td[i], mean_last(cd, 24),
        p[i], p[i] - p[i - 3], p[i] - p[i - 12], p[i] - p[i - 24], td[i] - td[i - 24],
        win.max(1) - win.min(1), np.sin(doy), np.cos(doy), np.sin(hr), np.cos(hr)])
    target = np.full((N, H), np.nan)
    has = idx + H <= N - 1
    target[has] = t[idx[has][:, None] + h[None, :]]
    return base, feats, target


def check_equivalence(train, base, feats):
    """PLAN §10.4 check 1: at 23:00 origins, hourly M1 and first 37 features equal forecast.py's."""
    T = F.day_matrix(train, 'temperature_c')
    d_base, d_feat = F.m1_all(T, K, 1, ALPHA_DECAY), F.origin_features(train, K)
    ends = np.arange(len(T)) * 24 + 23
    rows = np.isfinite(d_feat).all(1) & np.isfinite(d_base).all(1) & (ends >= FIRST)
    assert np.allclose(base[ends[rows]], d_base[rows], atol=1e-9, rtol=0)
    assert np.allclose(feats[ends[rows], :37], d_feat[rows], atol=1e-9, rtol=0)
    return int(rows.sum())


def predict(base, feats, target, c, alpha, daily_only=False):
    """Fit on origins i with FIRST <= i and i + 72 <= c; predict origin c."""
    i = np.arange(FIRST, c - H + 1)
    if daily_only:
        i = i[i % 24 == 23]
    assert i.max() + H <= c                                       # check 2: all targets end by c
    X = feats[i][:, :37] if daily_only else feats[i]
    x0 = feats[c][:37] if daily_only else feats[c]
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(X, target[i] - base[i])
    return base[c] + model.predict(x0[None])[0], len(i)


def main():
    train, _ = load_data()
    ts = train.timestamp
    base, feats, target = hourly_tables(train)
    n_eq = check_equivalence(train, base, feats)
    print(f'check 1: 23:00 rows equal to forecast.py (M1 + 37 features): {n_eq} rows OK')
    pos = {t: k for k, t in enumerate(ts)}
    cands = [('daily_ridge1', 1, True)] + [(f'hourly_ridge{a}', a, False) for a in RIDGE_ALPHAS]
    rows = []
    for o in A + B:
        c = pos[o]
        y = train.temperature_c.to_numpy()[c + 1: c + H + 1]
        for name, a, daily in cands:
            pred, n = predict(base, feats, target, c, a, daily)
            e = np.abs(pred - y)
            rows.append(dict(candidate=name, origin=str(o), set='A1' if o in A1 else 'A2' if o in A2 else 'B',
                             n_train=n, mae=e.mean(), h1_24=e[:24].mean(), h25_48=e[24:48].mean(), h49_72=e[48:].mean()))
    # check 3: corrupting everything after an origin changes neither features nor the prediction
    for o in [A[0], A[-1], B[-1]]:
        c = pos[o]
        bad = train.copy()
        cols = [bad.columns.get_loc(k) for k in ('temperature_c', 'dew_point_c', 'pressure_hpa')]
        bad.iloc[c + 1:, cols] = 999.0
        b2, f2, t2 = hourly_tables(bad)
        assert np.array_equal(np.nan_to_num(f2[:c + 1]), np.nan_to_num(feats[:c + 1]))
        for a in RIDGE_ALPHAS:
            assert np.allclose(predict(b2, f2, t2, c, a)[0], predict(base, feats, target, c, a)[0], atol=1e-9)
    print('check 3: corruption after origin leaves features and predictions unchanged OK')
    res = pd.DataFrame(rows)
    res.to_csv('logs/hourly_results.csv', index=False)

    ref = res[res.candidate == 'daily_ridge1'].set_index('origin')
    print(f"integration check: daily_ridge1 A = {ref[ref.set != 'B'].mae.mean():.3f} (record 2.382), "
          f"B = {ref[ref.set == 'B'].mae.mean():.3f} (record 2.151)")
    first = res[res.origin == str(A[0])].set_index('candidate').n_train
    print(f'check 4: training rows at first A origin: daily {first["daily_ridge1"]}, hourly {first["hourly_ridge1"]} '
          f'(expected hourly = {pos[A[0]] - H - FIRST + 1})')
    print('\ncandidate        A      A1     A2     B    h1-24  h25-48 h49-72  dA_vs_daily  winA   t_p    wil_p  adopt')
    for name, _, _ in cands:
        g = res[res.candidate == name].set_index('origin')
        a = g[g.set != 'B']
        dA = a.mae - ref.loc[a.index].mae
        dB = g[g.set == 'B'].mae.mean() - ref[ref.set == 'B'].mae.mean()
        win = (dA < 0).mean()
        if name == 'daily_ridge1':
            tp = wp = np.nan
        else:
            tp, wp = ttest_rel(a.mae, ref.loc[a.index].mae).pvalue, wilcoxon(dA).pvalue
        adopt = name != 'daily_ridge1' and dA.mean() <= -0.05 and win >= 0.6 and dB <= 0.1
        print(f'{name:15s} {a.mae.mean():.3f} {g[g.set == "A1"].mae.mean():.3f} {g[g.set == "A2"].mae.mean():.3f} '
              f'{g[g.set == "B"].mae.mean():.3f}  {a.h1_24.mean():.3f}  {a.h25_48.mean():.3f}  {a.h49_72.mean():.3f}  '
              f'{dA.mean():+.3f}  {win:5.1%}  {tp:.3f}  {wp:.3f}  {adopt}')


if __name__ == '__main__':
    main()
