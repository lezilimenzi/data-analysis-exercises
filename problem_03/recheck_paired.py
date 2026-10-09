"""Paired-difference statistics for STRATEGY.md grades (added 2026-10-02).
Per validation origin: diff = baseline MAE - candidate MAE (positive = candidate better).
Uses forecast.py and saved per-origin results (logs/hourly_results.csv, logs/m4_lstm_h*.csv, logs/hourly_mix_C.csv).
Output: logs/paired_recheck.csv
"""
import numpy as np
import pandas as pd

import forecast as F
from starter import load_data

train, _ = load_data()
temp = train.set_index('timestamp').temperature_c


def origins(a, b):
    return list(pd.date_range(f'{a} 23:00', f'{b} 23:00', freq='3D'))


A = origins('2013-11-01', '2014-02-25') + origins('2014-11-01', '2014-11-28')
keys = [str(o) for o in A]


def errors(fn):
    out = []
    for o in A:
        h = train[train.timestamp <= o]
        y = temp[o + pd.Timedelta(hours=1): o + pd.Timedelta(hours=72)].to_numpy()
        out.append(np.abs(fn(h) - y))
    return np.array(out)


e_hm5, e_m1 = errors(lambda h: F.hour_mean(h, 5)), errors(lambda h: F.m1(h, 5, 1, 0.97))
res = pd.read_csv('logs/hourly_results.csv')
ridge = res[(res.candidate == 'daily_ridge1') & (res.set != 'B')].set_index('origin').mae.loc[keys].to_numpy()
assert abs(ridge.mean() - 2.382) < 0.0015 and abs(e_m1.mean() - 2.671) < 0.0015

rows = {
    'A: hm5 -> M1 (72h)': e_hm5.mean(1) - e_m1.mean(1),
    'A: hm5 -> M1 (first 24h)': e_hm5[:, :24].mean(1) - e_m1[:, :24].mean(1),
    'A: M1 -> M2-ridge': e_m1.mean(1) - ridge,
    'A: hm5 -> M2-ridge': e_hm5.mean(1) - ridge,
}
for h in (32, 64):
    lstm = pd.read_csv(f'logs/m4_lstm_h{h}.csv', index_col=0).mean(1).loc[keys].to_numpy()
    rows[f'A: M2-ridge -> LSTM h{h}'] = ridge - lstm
mix = pd.read_csv('logs/hourly_mix_C.csv')
for s in ('C', 'C0'):
    m = mix[mix.set == s]
    base = m[m.candidate == 'daily_ridge1'].set_index('origin').mae
    for cand in m.candidate.unique():
        if cand != 'daily_ridge1':
            rows[f'{s}: daily_ridge1 -> {cand}'] = (base - m[m.candidate == cand].set_index('origin').mae).to_numpy()

out = pd.DataFrame([dict(comparison=k, n=len(d), mean=d.mean(), sd=d.std(ddof=1), mean_over_sd=d.mean() / d.std(ddof=1),
                         better=int((d > 0).sum()), worse=int((d < 0).sum())) for k, d in rows.items()])
out.to_csv('logs/paired_recheck.csv', index=False)
print(out.round(3).to_string(index=False))
