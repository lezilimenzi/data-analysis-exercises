"""Post-hoc recheck (2026-09-28): full M1 grid on validation set A, saved for STRATEGY.md 6.3.
The original stage 'm1' in evaluate.py printed only the top 10. Same origins, same forecast.py.
Output: logs/m1_grid_recheck.csv (model, k, m, alpha, A_mae). Existing files are not modified.
"""
import numpy as np
import pandas as pd

import forecast as F
from starter import load_data

TRAIN, _ = load_data()
TEMP = TRAIN.set_index('timestamp').temperature_c


def origins(a, b):
    return list(pd.date_range(f'{a} 23:00', f'{b} 23:00', freq='3D'))


A = origins('2013-11-01', '2014-02-25') + origins('2014-11-01', '2014-11-28')  # same as evaluate.py
CASES = [(TRAIN[TRAIN.timestamp <= o], TEMP[o + pd.Timedelta(hours=1): o + pd.Timedelta(hours=72)].to_numpy()) for o in A]
assert len(A) == 49 and all(len(y) == 72 for _, y in CASES)


def a_mae(fn):
    return float(np.mean([np.abs(fn(h) - y).mean() for h, y in CASES]))


rows = []
for k in [2, 3, 4, 5, 7]:
    rows.append(dict(model='hm', k=k, m=np.nan, alpha=np.nan, A_mae=a_mae(lambda h: F.hour_mean(h, k))))
    for m in [1, 3, 6]:
        for alpha in [0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99]:
            rows.append(dict(model='M1', k=k, m=m, alpha=alpha, A_mae=a_mae(lambda h: F.m1(h, k, m, alpha))))
rows.append(dict(model='M1_no_decay', k=5, m=1, alpha=1.0, A_mae=a_mae(lambda h: F.m1(h, 5, 1, 1.0))))

out = pd.DataFrame(rows)
grid = out[out.model == 'M1'].sort_values('A_mae')
assert len(grid) == 105
best = grid.iloc[0]
assert (best.k, best.m, best.alpha) == (5, 1, 0.97) and round(best.A_mae, 3) == 2.671  # original record
out.to_csv('logs/m1_grid_recheck.csv', index=False)
print(grid.head(5).round(3).to_string(index=False))
print(out[out.model != 'M1'].round(3).to_string(index=False))

# Per-horizon A MAE of hm5 vs the selected M1 (added 2026-10-02 for STRATEGY.md 6.3 evidence table).
hz = []
for name, fn in [('hm5', lambda h: F.hour_mean(h, 5)), ('M1', lambda h: F.m1(h, 5, 1, 0.97))]:
    e = np.array([np.abs(fn(h) - y) for h, y in CASES])
    hz.append(dict(model=name, h1_24=e[:, :24].mean(), h25_48=e[:, 24:48].mean(), h49_72=e[:, 48:].mean(), all=e.mean()))
hz = pd.DataFrame(hz)
assert abs(hz.loc[hz.model == 'M1', 'h1_24'].item() - 2.054) < 0.0015  # RESEARCH §8.1 record
hz.to_csv('logs/m1_horizon_recheck.csv', index=False)
print(hz.round(3).to_string(index=False))
