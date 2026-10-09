"""PLAN §11: H1 horizon mix and H2 origin-hour weighting, tuned on A, judged ONCE on the fresh set C.
Reuses exp_hourly.hourly_tables (checked equal to forecast.py at 23:00). Existing files are not modified.
Outputs: logs/hourly_mix_A.csv (tuning), logs/hourly_mix_C.csv (C, C0, B evaluation).
"""
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from exp_hourly import A, A1, A2, B, FIRST, H, hourly_tables, origins
from starter import load_data

H0S, SIGMAS, HOURLY_ALPHA = [24, 48], [1, 3, 6], 10
C = origins('2014-03-01', '2014-10-28')
C0 = origins('2013-03-01', '2013-10-28')


def fit_pred(base, feats, target, c, alpha, daily=False, sigma=None):
    """Ridge on M1 residuals from origins i (FIRST <= i, i + 72 <= c); predict origin c."""
    i = np.arange(FIRST, c - H + 1)
    if daily:
        i = i[i % 24 == 23]  # train starts at 00:00, so index % 24 is the clock hour
    assert i.max() + H <= c
    cols = slice(0, 37) if daily else slice(None)
    w = None
    if sigma is not None:
        d = np.abs(i % 24 - 23)
        d = np.minimum(d, 24 - d)  # circular distance to 23:00
        w = np.exp(-d ** 2 / (2 * sigma ** 2))
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(feats[i][:, cols], target[i] - base[i], ridge__sample_weight=w)
    return base[c] + model.predict(feats[c][None, cols])[0]


def err_row(name, o, set_, e):
    return dict(candidate=name, origin=str(o), set=set_, mae=e.mean(), h1_24=e[:24].mean(),
                h25_48=e[24:48].mean(), h49_72=e[48:].mean())


def evaluate(train, tabs, pos, origins_, set_name, sigmas, h0s):
    temp = train.temperature_c.to_numpy()
    rows = []
    for o in origins_:
        c = pos[o]
        y = temp[c + 1: c + H + 1]
        assert len(y) == H
        p_day = fit_pred(*tabs, c, 1, daily=True)
        p_hr = fit_pred(*tabs, c, HOURLY_ALPHA)
        s = set_name(o)
        rows += [err_row('daily_ridge1', o, s, np.abs(p_day - y)), err_row('hourly_ridge10', o, s, np.abs(p_hr - y))]
        for h0 in h0s:
            p = np.concatenate([p_day[:h0], p_hr[h0:]])
            rows.append(err_row(f'H1_mix{h0}', o, s, np.abs(p - y)))
        for sg in sigmas:
            rows.append(err_row(f'H2_sigma{sg}', o, s, np.abs(fit_pred(*tabs, c, HOURLY_ALPHA, sigma=sg) - y)))
    return pd.DataFrame(rows)


def report(res, ref_name, set_):
    ref = res[(res.candidate == ref_name) & (res.set == set_)].set_index('origin')
    print(f'\n[{set_}] n={len(ref)}  candidate        MAE    1-24h  25-48h 49-72h  diff    win    t_p    wil_p')
    out = {}
    for name in res.candidate.unique():
        g = res[(res.candidate == name) & (res.set == set_)].set_index('origin')
        d = g.mae - ref.loc[g.index].mae
        tp = wp = np.nan
        if name != ref_name:
            tp, wp = ttest_rel(g.mae, ref.loc[g.index].mae).pvalue, wilcoxon(d).pvalue
        out[name] = (d.mean(), (d < 0).mean())
        print(f'{"":16s}{name:16s} {g.mae.mean():.3f}  {g.h1_24.mean():.3f}  {g.h25_48.mean():.3f}  '
              f'{g.h49_72.mean():.3f}  {d.mean():+.3f}  {(d < 0).mean():5.1%}  {tp:.3f}  {wp:.3f}')
    return out


def main():
    train, _ = load_data()
    tabs = hourly_tables(train)
    pos = {t: k for k, t in enumerate(train.timestamp)}
    # C / C0 72h windows must not overlap A or B evaluation hours
    assert max(C) + pd.Timedelta(hours=H) < min(A2) + pd.Timedelta(hours=1)
    assert max(C0) + pd.Timedelta(hours=H) < min(A1) + pd.Timedelta(hours=1)

    # 1) tuning on A (already-seen set)
    resA = evaluate(train, tabs, pos, A, lambda o: 'A1' if o in A1 else 'A2', SIGMAS, H0S)
    resA.to_csv('logs/hourly_mix_A.csv', index=False)
    mA = resA.groupby('candidate').mae.mean()
    assert abs(mA['daily_ridge1'] - 2.382) < 0.0015 and abs(mA['hourly_ridge10'] - 2.373) < 0.0015
    print('A tuning (mean MAE):'); print(mA.round(4).to_string())
    h0 = min(H0S, key=lambda v: (round(mA[f'H1_mix{v}'], 10), H0S.index(v)))
    sg = min(SIGMAS, key=lambda v: (round(mA[f'H2_sigma{v}'], 10), SIGMAS.index(v)))
    print(f'selected on A: H1 H0={h0}, H2 sigma={sg}')
    # check: very large sigma == unweighted hourly ridge
    for o in [A[0], A[-1]]:
        c = pos[o]
        assert np.allclose(fit_pred(*tabs, c, HOURLY_ALPHA, sigma=1e6), fit_pred(*tabs, c, HOURLY_ALPHA), atol=1e-6)
    print('check: sigma=1e6 equals unweighted hourly ridge OK')

    # 2) judged ONCE on C (fresh); C0 and B are reference only
    lab = lambda o: 'C' if o in C else 'C0' if o in C0 else 'B'
    res = evaluate(train, tabs, pos, C + C0 + B, lab, [sg], [h0])
    res.to_csv('logs/hourly_mix_C.csv', index=False)
    stats = report(res, 'daily_ridge1', 'C')
    report(res, 'daily_ridge1', 'C0')
    report(res, 'daily_ridge1', 'B')
    print('\nJudgement on C (PLAN §11.4: diff <= -0.05 and win >= 60%):')
    passed = []
    for name in [f'H1_mix{h0}', f'H2_sigma{sg}']:
        dm, win = stats[name]
        ok = dm <= -0.05 and win >= 0.6
        passed.append(ok)
        print(f'  {name}: diff {dm:+.3f}, win {win:.1%} -> adopt={ok}')
    print(f'H3 (hourly GBM) proceeds: {any(passed)}')


if __name__ == '__main__':
    main()
