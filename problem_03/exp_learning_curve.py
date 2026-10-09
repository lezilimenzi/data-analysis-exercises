"""PLAN §10.3 stage 1: learning curve of M2 residual models on the existing 23:00 samples.
For every A origin, train on a random subset (25/50/75/100%) of the rows forecast.residual_data gives,
then score the 72h forecast. Ridge and gbm use identical subsets. Results are saved per origin
(logs/learning_curve.csv) so a re-run skips finished (model, origin) pairs. Existing files untouched.
Usage: python3 exp_learning_curve.py [--summary]
"""
import os, sys, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import forecast as F
from starter import load_data

OUT = 'logs/learning_curve.csv'
FRACS = [0.25, 0.5, 0.75, 1.0]
REPS = [0, 1, 2]
GBM_BUDGET_S = 3600  # PLAN §10.3: if projected gbm time exceeds 60 min, use 1 subsample draw per fraction


def origins(a, b):
    return list(pd.date_range(f'{a} 23:00', f'{b} 23:00', freq='3D'))


A1 = origins('2013-11-01', '2014-02-25')
A2 = origins('2014-11-01', '2014-11-28')
A = A1 + A2  # same as evaluate.py


def fit_predict(model, X, R, x0, b0):
    if model == 'ridge':  # = forecast.m2_ridge with M2_BEST
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X, R)
        return b0 + m.predict(x0[None])[0]
    m = HistGradientBoostingRegressor(  # = forecast.m2_gbm(loss='squared_error', max_iter=200, max_leaf_nodes=31)
        loss='squared_error', max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=50, early_stopping=False, random_state=0)
    m.fit(F.stack_horizons(X), R.ravel())
    return b0 + m.predict(F.stack_horizons(x0[None]))


def draws(n, oi):
    """Row subsets per (fraction, rep); identical for both models. 100% has a single draw."""
    for frac in FRACS:
        for rep in (REPS if frac < 1 else [0]):
            if frac == 1:
                yield frac, rep, np.arange(n)
            else:
                rng = np.random.default_rng(10_000 * rep + oi)
                yield frac, rep, np.sort(rng.choice(n, size=int(round(frac * n)), replace=False))


def run():
    train, _ = load_data()
    temp = train.set_index('timestamp').temperature_c
    done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame()
    reps_gbm = REPS
    for model in ['ridge', 'gbm']:
        for oi, o in enumerate(A):
            if len(done) and ((done.model == model) & (done.origin == str(o))).any():
                continue
            t0 = time.time()
            hist = train[train.timestamp <= o]
            assert hist.timestamp.iloc[-1] == o
            y = temp[o + pd.Timedelta(hours=1): o + pd.Timedelta(hours=72)].to_numpy()
            assert len(y) == 72
            X, R, x0, b0 = F.residual_data(hist)  # rows whose 72 targets all lie inside hist (asserted there)
            rows = []
            for frac, rep, idx in draws(len(X), oi):
                if model == 'gbm' and frac < 1 and rep not in reps_gbm:
                    continue
                e = np.abs(fit_predict(model, X[idx], R[idx], x0, b0) - y)
                rows.append(dict(model=model, origin=str(o), set='A1' if o in A1 else 'A2', frac=frac, rep=rep,
                                 n_train=len(idx), mae=e.mean(), h1_24=e[:24].mean(), h25_48=e[24:48].mean(),
                                 h49_72=e[48:].mean()))
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            done.to_csv(OUT, index=False)
            sec = time.time() - t0
            print(f'{model} {o} n={len(X)} fits={len(rows)} {sec:.1f}s', flush=True)
            if model == 'gbm' and oi == 0 and sec * len(A) > GBM_BUDGET_S:
                reps_gbm = [0]
                print(f'  gbm projected {sec * len(A) / 60:.0f} min > 60 -> 1 draw per fraction from next origin '
                      f'(PLAN §10.3)', flush=True)


def summary():
    d = pd.read_csv(OUT)
    ref = {'ridge': 2.382, 'gbm': 2.532}  # original A records (RESEARCH §8.1)
    for model, g in d.groupby('model'):
        full = g[g.frac == 1].mae.mean()
        print(f'\n== {model}: 100% A = {full:.3f} (original record {ref[model]}) '
              f'{"OK" if abs(full - ref[model]) < 0.0015 else "MISMATCH"}')
        per_rep = g.groupby(['frac', 'rep']).agg(A=('mae', 'mean'), n=('n_train', 'mean'))
        tab = per_rep.groupby('frac').agg(A_mean=('A', 'mean'), A_sd=('A', 'std'), draws=('A', 'size'), n_mean=('n', 'mean'))
        for s in ['A1', 'A2']:
            tab[s] = g[g.set == s].groupby('frac').mae.mean()
        for h in ['h1_24', 'h25_48', 'h49_72']:
            tab[h] = g.groupby('frac')[h].mean()
        print(tab.round(3).to_string())
        a = tab.A_mean
        drop_75_100 = a[0.75] - a[1.0]
        ok = drop_75_100 >= 0.02 and a[0.75] < a[0.5]
        print(f'75%->100% improvement {drop_75_100:+.3f}, 50%->75% {a[0.5] - a[0.75]:+.3f} '
              f'-> stage 2 for {model}: {ok}  (rule: >= 0.02 and 50->75 improves; PLAN §10.3)')


if __name__ == '__main__':
    if sys.argv[1:] != ['--summary']:
        run()
    summary()
