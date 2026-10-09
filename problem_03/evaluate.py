"""Rolling-origin validation. Usage: python evaluate.py <stage>  (m0, m1, ...)"""
import sys
from functools import partial
from itertools import product

import numpy as np
import pandas as pd

import forecast as F
from starter import load_data

TRAIN, _ = load_data()
TEMP = TRAIN.set_index('timestamp').temperature_c


def origins(a, b):
    return list(pd.date_range(f'{a} 23:00', f'{b} 23:00', freq='3D'))


A1 = origins('2013-11-01', '2014-02-25')
A2 = origins('2014-11-01', '2014-11-28')
A = A1 + A2
B = origins('2014-12-01', '2014-12-25')
REC3 = [pd.Timestamp(f'2014-12-{d} 23:00') for d in (19, 22, 25)]
SETS = {'A': A, 'A1': A1, 'A2': A2, 'B': B, 'REC3': REC3}


def run(fn, origs):
    """Absolute errors (folds, 72). The model only ever sees data up to each origin."""
    errs = []
    for o in origs:
        hist = TRAIN[TRAIN.timestamp <= o]
        assert hist.timestamp.iloc[-1] == o
        y = TEMP[o + pd.Timedelta(hours=1): o + pd.Timedelta(hours=72)].to_numpy()
        assert len(y) == 72
        p = np.asarray(fn(hist), float)
        assert p.shape == (72,) and np.isfinite(p).all()
        errs.append(np.abs(p - y))
    return pd.DataFrame(errs, index=origs)


def summarize(results, ref=None):
    """results: {name: errs}. Per-set mean MAE, horizon buckets on A, paired stats vs ref on A."""
    rows = {}
    for name, e in results.items():
        mae = e.mean(1)
        r = {s: mae[mae.index.isin(o)].mean() for s, o in SETS.items() if mae.index.isin(o).any()}
        ea = e[e.index.isin(A)]
        r.update({'h1-24': ea.iloc[:, :24].values.mean(), 'h25-48': ea.iloc[:, 24:48].values.mean(),
                  'h49-72': ea.iloc[:, 48:].values.mean()})
        if ref is not None and name != ref:
            d = (mae - results[ref].mean(1))[mae.index.isin(A)]
            r.update({'dA': d.mean(), 'winA%': 100 * (d < 0).mean()})
        rows[name] = r
    return pd.DataFrame(rows).T.round(3)


def grid(make_fn, space, origs=A):
    """A-set mean MAE for every parameter combination, sorted."""
    keys = list(space)
    out = []
    for vals in product(*space.values()):
        params = dict(zip(keys, vals))
        out.append({**params, 'A': run(make_fn(**params), origs).values.mean()})
    return pd.DataFrame(out).sort_values('A').reset_index(drop=True)


pd.set_option('display.width', 200)
pd.set_option('display.max_columns', None)
stage = sys.argv[1]

if stage == 'm0':
    res = {'persist': run(F.persist, A + B)}
    for k in (1, 3, 5, 7):
        res[f'hm{k}'] = run(partial(F.hour_mean, k=k), A + B)
    print(summarize(res, ref='hm5'))

elif stage == 'm1':
    g = grid(lambda **p: partial(F.m1, **p),
             {'k': [2, 3, 4, 5, 7], 'm': [1, 3, 6], 'alpha': [0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99]})
    print('M1 grid (A), top 10 of', len(g)); print(g.head(10).round(3))
    gb = grid(lambda **p: partial(F.m1, **p),
              {'k': [7, 14], 'k_level': [1, 2, 3], 'm': [1, 3, 6], 'alpha': [0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99]})
    print('\nM1b grid (A), top 10 of', len(gb)); print(gb.head(10).round(3))
    best = g.iloc[0].drop('A').to_dict(); best_b = gb.iloc[0].drop('A').to_dict()
    for p in (best, best_b):
        for key in ('k', 'm', 'k_level'):
            if key in p: p[key] = int(p[key])
    print('\nbest M1:', best, '\nbest M1b:', best_b)
    res = {'hm5': run(partial(F.hour_mean, k=5), A),
           'M1': run(partial(F.m1, **best), A),
           'M1b': run(partial(F.m1, **best_b), A)}
    print(summarize(res, ref='hm5'))
    print('\nM1b vs M1:'); print(summarize({'M1': res['M1'], 'M1b': res['M1b']}, ref='M1').loc[['M1b']])

elif stage == 'm2':
    m1_best = partial(F.m1, **F.M1_BEST)
    g = grid(lambda **p: partial(F.m2_ridge, **p), {'alpha': [0.1, 1, 10, 100], 'season_days': [None, 60]})
    print('M2-ridge grid (A)'); print(g.round(3), flush=True)
    gg = grid(lambda **p: partial(F.m2_gbm, **p),
              {'loss': ['squared_error', 'absolute_error'], 'max_iter': [200, 500],
               'max_leaf_nodes': [15, 31], 'season_days': [None, 60]})
    print('\nM2-gbm grid (A)'); print(gg.round(3), flush=True)
    best_r = {'alpha': g.alpha[0], 'season_days': g.season_days[0]}
    best_g = {k: gg[k][0] for k in ('loss', 'max_iter', 'max_leaf_nodes', 'season_days')}
    best_g['max_iter'], best_g['max_leaf_nodes'] = int(best_g['max_iter']), int(best_g['max_leaf_nodes'])
    for b in (best_r, best_g):
        if pd.isna(b['season_days']): b['season_days'] = None
    print('\nbest ridge:', best_r, '\nbest gbm:', best_g)
    res = {'M1': run(m1_best, A), 'M2-ridge': run(partial(F.m2_ridge, **best_r), A),
           'M2-gbm': run(partial(F.m2_gbm, **best_g), A)}
    print(summarize(res, ref='M1'))

elif stage == 'm4':
    for kind, hidden in (('mlp', [64, 128]), ('lstm', [32, 64])):
        g = grid(lambda **p: partial(F.m4, **p), {'kind': [kind], 'hidden': hidden})
        print(f'M4-{kind} grid (A)'); print(g.round(3), flush=True)

elif stage == 'lstm':  # rerun of the M4-lstm half with per-fold progress; errors saved for paired comparison
    import time
    for hidden in (32, 64):
        t0 = time.time()

        def fn(hist, hidden=hidden):
            p = F.m4(hist, kind='lstm', hidden=hidden)
            print(f'  hidden={hidden} {hist.timestamp.iloc[-1]} done ({time.time() - t0:.0f}s)', flush=True)
            return p

        e = run(fn, A)
        e.to_csv(f'logs/m4_lstm_h{hidden}.csv')
        print(f'M4-lstm hidden={hidden}: A {e.values.mean():.3f}', flush=True)

elif stage == 'm3':
    res = {'M2-ridge': run(partial(F.m2_ridge, **F.M2_BEST), A)}
    for w in (0.3, 0.5, 0.7):
        res[f'M3 w={w}'] = run(partial(F.m3, w=w), A)
    print(summarize(res, ref='M2-ridge'))

elif stage == 'holdout':  # step 6: final candidate vs simpler models on B (not used for tuning)
    res = {'hm5': run(partial(F.hour_mean, k=5), B), 'M1': run(partial(F.m1, **F.M1_BEST), B),
           'M2-ridge': run(partial(F.m2_ridge, **F.M2_BEST), B)}
    print(summarize(res, ref='M1').drop(columns=['h1-24', 'h25-48', 'h49-72', 'dA', 'winA%']))
    mae = {k: e.mean(1) for k, e in res.items()}
    print('\nper-fold B MAE'); print(pd.DataFrame(mae).round(2))
