"""Post-hoc strategy study (2026-10-06): candidates pre-registered before any result was seen.

Pairs: every 23:00 origin 2014-01-01..2014-12-25 (359). Baseline: final M2-ridge.
d_i = baseline MAE - candidate MAE (positive = candidate better), ties |d| <= 1e-9.
Adopt-eligible: mean d > 0 and wins > losses (maicon/CLAUDE.md section 5). Strategy learning only:
no new submission, no test/instructor data. Existing files are not modified.
Outputs: logs/strategy_pairs.csv (per-origin MAE + diagnostics), logs/strategy.out (summary).
"""
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import forecast as F
from starter import load_data

H = F.H
ORIGINS = list(pd.date_range('2014-01-01 23:00', '2014-12-25 23:00', freq='D'))
COST_ORDER = ['S1', 'F3', 'F1', 'F2', 'F-all', 'S2', 'S3', 'S5']  # pre-registered cost priority
ANALOG_K, ANALOG_SEASON = 15, 45


# ---------------- derived feature groups (all known at the end of day d) ----------------
def extra_features(hist, group):
    T, Td, P = (F.day_matrix(hist, c) for c in ('temperature_c', 'dew_point_c', 'pressure_hpa'))
    N = len(T)
    p = P.ravel()
    end = np.arange(N) * 24 + 23

    def shift(x, n):  # x[d - n] (days), NaN where unavailable
        out = np.full(x.shape, np.nan)
        out[n:] = x[:N - n]
        return out

    def plag(n):
        out = np.full(N, np.nan)
        ok = end - n >= 0
        out[ok] = p[end[ok] - n]
        return out

    if group == 'F1':  # humidity
        es = lambda t: np.exp(17.625 * t / (243.04 + t))  # Magnus
        rh = 100 * es(Td[:, 23]) / es(T[:, 23])
        return np.column_stack([rh, Td.min(1), Td.max(1), T[:, 14] - Td[:, 14]])
    if group == 'F2':  # diurnal-range persistence (cloud proxy)
        dtr = T.max(1) - T.min(1)
        dtr3 = (dtr + shift(dtr, 1) + shift(dtr, 2)) / 3
        night_cool = T[:, 6] - shift(T[:, 18], 1)
        morning_heat = T[:, 14] - T[:, 6]
        return np.column_stack([dtr3, night_cool, morning_heat])
    if group == 'F3':  # pressure regime
        c = np.concatenate([[0], np.cumsum(p)])
        p30 = np.full(N, np.nan)
        ok = end + 1 >= 720
        p30[ok] = (c[end[ok] + 1] - c[end[ok] + 1 - 720]) / 720
        accel = (p[end] - plag(3)) - (plag(12) - plag(15))
        return np.column_stack([p[end] - plag(48), p[end] - plag(72), accel, p[end] - p30,
                                (p[end] - plag(24) <= -5).astype(float)])
    raise ValueError(group)


def daily_tables(hist):
    """Per origin day: M1 base (72), 37 base features, actual next 72h (rows d <= N-4)."""
    T = F.day_matrix(hist, 'temperature_c')
    N = len(T)
    base = F.m1_all(T, **F.M1_BEST)
    X = F.origin_features(hist, F.M1_BEST['k'])
    n = N - 3
    Y = np.hstack([T[1:n + 1], T[2:n + 2], T[3:n + 3]])
    return T, base, X, Y, n


def ridge_fit_pred(X, R, x0):
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X, R).predict(x0[None])[0]


def train_rows(*arrays, n):
    ok = np.ones(n, bool)
    for a in arrays:
        ok &= np.isfinite(a[:n].reshape(n, -1)).all(1)
    return np.flatnonzero(ok)


# ---------------- candidates ----------------
def cand_ridge(tabs, extra=None):
    T, base, X, Y, n = tabs
    Xf = X if extra is None else np.hstack([X, extra])
    i = train_rows(Xf, base, n=n)
    assert i.max() + 3 <= len(T) - 1
    return base[-1] + ridge_fit_pred(Xf[i], (Y - base[:n])[i], Xf[-1])


def cand_s1(tabs):
    """Median (L1) regression per horizon on the same 37 features; alpha fixed a priori."""
    T, base, X, Y, n = tabs
    i = train_rows(X, base, n=n)
    sc = StandardScaler().fit(X[i])
    Z, z0 = sc.transform(X[i]), sc.transform(X[-1:])
    R = (Y - base[:n])[i]
    return base[-1] + np.array([QuantileRegressor(quantile=0.5, alpha=0.001, solver='highs')
                                .fit(Z, R[:, h]).predict(z0)[0] for h in range(H)])


def cand_s2(tabs):
    """Level + amplitude x 5-day shape per forecast day, on top of M1's decaying anomaly."""
    T, base, X, Y, n = tabs
    prof = F.profile(T, F.M1_BEST['k'])
    shape = prof - prof.mean(1, keepdims=True)  # (days, 24)
    lvl0 = prof.mean(1)
    decay = base - np.tile(prof, 3)  # anomaly * alpha^h part of M1
    resid = (Y - lvl0[:n, None] - decay[:n]).reshape(n, 3, 24)
    lev = resid.mean(2)  # (n, 3)
    s = shape[:n, None, :]
    amp = ((resid - lev[..., None]) * s).sum(2) / (s ** 2).sum(2)  # projection on shape
    targets = np.hstack([lev, amp])
    i = train_rows(X, targets, n=n)
    out = ridge_fit_pred(X[i], targets[i], X[-1])
    L, A_ = out[:3], out[3:]
    return lvl0[-1] + decay[-1] + np.repeat(L, 24) + np.tile(shape[-1], 3) * np.repeat(A_, 24)


def analog_inputs(hist):
    """Per origin day: last 72h (every 3h) of T anomaly vs profile_d, T-Td, P - P_now, plus P_now."""
    T, Td, P = (F.day_matrix(hist, c) for c in ('temperature_c', 'dew_point_c', 'pressure_hpa'))
    N = len(T)
    prof = F.profile(T, F.M1_BEST['k'])
    t, td, p = T.ravel(), Td.ravel(), P.ravel()
    out = np.full((N, 3 * 24 + 1), np.nan)
    for d in range(2, N):
        s = slice((d - 2) * 24, (d + 1) * 24, 3)
        e = (d + 1) * 24 - 1
        out[d] = np.concatenate([t[s] - np.tile(prof[d], 3)[::3], t[s] - td[s], p[s] - p[e], [p[e]]])
    return out


def cand_s3(tabs, hist):
    T, base, X, Y, n = tabs
    V = analog_inputs(hist)
    R = Y - base[:n]
    i = train_rows(V, base, n=n)
    assert i.max() + 3 <= len(T) - 1
    doy = hist.timestamp.dt.dayofyear.to_numpy()[23::24]
    dist = np.abs(doy[i] - doy[-1])
    i = i[np.minimum(dist, 365 - dist) <= ANALOG_SEASON]
    sd = V[i].std(0) + 1e-6
    D = np.sqrt((((V[i] - V[-1]) / sd) ** 2).sum(1))
    nn = i[np.argsort(D)[:ANALOG_K]]
    return base[-1] + np.median(R[nn], axis=0)


# ---------------- diagnostics ----------------
def diagnostics(pred, y):
    p3, y3 = pred.reshape(3, 24), y.reshape(3, 24)
    dtr_err = np.mean(np.abs(np.ptp(p3, 1) - np.ptp(y3, 1)))
    ev = [j for j in range(3) if y3[j, 19:24].max() - y3[j, 18] >= 2]  # nocturnal warming events
    ev_mae = np.mean([np.abs(p3[j, 18:] - y3[j, 18:]).mean() for j in ev]) if ev else np.nan
    return dtr_err, ev_mae


def main():
    train, _ = load_data()
    temp = train.set_index('timestamp').temperature_c
    rows, t0 = [], time.time()
    for k, o in enumerate(ORIGINS):
        hist = train[train.timestamp <= o]
        assert hist.timestamp.iloc[-1] == o
        y = temp[o + pd.Timedelta(hours=1): o + pd.Timedelta(hours=H)].to_numpy()
        assert len(y) == H
        tabs = daily_tables(hist)
        ex = {g: extra_features(hist, g) for g in ('F1', 'F2', 'F3')}
        preds = {'base': cand_ridge(tabs)}
        assert np.allclose(preds['base'], F.m2_ridge(hist, **F.M2_BEST))  # baseline == final model
        preds['S1'] = cand_s1(tabs)
        for g in ('F1', 'F2', 'F3'):
            preds[g] = cand_ridge(tabs, ex[g])
        preds['F-all'] = cand_ridge(tabs, np.hstack([ex['F1'], ex['F2'], ex['F3']]))
        preds['S2'] = cand_s2(tabs)
        preds['S3'] = cand_s3(tabs, hist)
        preds['S5'] = 0.5 * preds['base'] + 0.5 * preds['S3']
        for name, p in preds.items():
            assert p.shape == (H,) and np.isfinite(p).all()
            e = np.abs(p.astype(np.float64) - y)
            dtr, ev = diagnostics(p, y)
            rows.append(dict(candidate=name, origin=str(o), mae=e.mean(), h1_24=e[:24].mean(),
                             h25_48=e[24:48].mean(), h49_72=e[48:].mean(), dtr_err=dtr, event_mae=ev))
        print(f'{k + 1}/{len(ORIGINS)} {o} ({time.time() - t0:.0f}s)', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv('logs/strategy_pairs.csv', index=False)
    summarize(df)


def summarize(df):
    base = df[df.candidate == 'base'].set_index('origin')
    winter = base.index.str[5:7].isin(['01', '02', '11', '12'])
    out = []
    for name in ['base'] + COST_ORDER:
        c = df[df.candidate == name].set_index('origin')
        d = base.mae - c.mae
        tie = d.abs() <= 1e-9
        wins, losses = int((d > 1e-9).sum()), int((d < -1e-9).sum())
        out.append(dict(candidate=name, MAE=c.mae.mean(), winter=c.mae[winter].mean(),
                        h1_24=c.h1_24.mean(), h25_48=c.h25_48.mean(), h49_72=c.h49_72.mean(),
                        dtr_err=c.dtr_err.mean(), event_mae=c.event_mae.mean(),
                        mean_d=d.mean(), wins=wins, losses=losses, ties=int(tie.sum()),
                        eligible=name != 'base' and d.mean() > 0 and wins > losses))
    s = pd.DataFrame(out).set_index('candidate')
    pd.set_option('display.width', 220)
    pd.set_option('display.max_columns', None)
    print(f'\npairs={len(base)} (winter subset {int(winter.sum())}), event days present in '
          f'{int(base.event_mae.notna().sum())} origins')
    print(s.round(4))
    elig = s[s.eligible]
    if elig.empty:
        print('\nNo candidate eligible -> keep baseline (final M2-ridge).')
    else:
        best = elig.mean_d.max()
        tied = [c for c in COST_ORDER if c in elig.index and best - elig.mean_d[c] <= 1e-9]
        print(f'\nEligible: {list(elig.index)} -> selected {tied[0]} (largest mean d; cost order on ties)')


if __name__ == '__main__':
    main()
