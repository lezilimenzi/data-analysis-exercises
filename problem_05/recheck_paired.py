"""Paired fold differences for STRATEGY.md grades (added 2026-10-02).
Re-runs the notebook's stage-1/2 models on the same folds (RepeatedStratifiedKFold 5x3, seed 42) using cached
features, and the stage-3 feature model on the same 3-fold. Fine-tuning fold scores are read from the saved
notebook output (they are the same TTA Macro-F1 the notebook used). The ensemble is NOT covered: its fold
probabilities were not saved. Asserts that recomputed means match cache/results.csv.
Output: paired_recheck.csv
"""
import json
import re
import warnings

import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from feature_helper import extract_features

warnings.filterwarnings('ignore')
SEED = 42; LABELS = [0, 1, 2, 3, 4]
tr = pd.read_csv('data/train.csv', encoding='utf-8-sig')
y = tr.label.values
rec = pd.read_csv('cache/results.csv').set_index('cand')
mf1 = lambda a, b: f1_score(a, b, labels=LABELS, average='macro', zero_division=0)


def color_stats(X):  # identical to notebook cell 1
    a = X.astype(np.float32); g = a.mean(-1)
    return np.c_[a[..., 0].mean((1, 2)), a[..., 1].mean((1, 2)), a[..., 2].mean((1, 2)), g.mean((1, 2)), a.std((1, 2, 3)),
                 (2 * a[..., 1] - a[..., 0] - a[..., 2]).mean((1, 2)),
                 np.abs(np.diff(g, axis=1)).mean((1, 2)) + np.abs(np.diff(g, axis=2)).mean((1, 2))]


def load(p):
    with Image.open(p) as im:
        return np.asarray(im.convert('RGB'))


CS = color_stats(np.stack([load(p) for p in tr.image_path]))
F = extract_features(tr.image_path, cache_path='cache/feat_orig_train.npz', verbose=False)
views = [list(tr.image_path)] + [[Path(f'cache/views/{k}/{i}.png') for i in tr.image_id] for k in range(1, 8)]
D = np.mean([extract_features(v, cache_path=f'cache/feat_train_v{k}.npz', verbose=False) for k, v in enumerate(views)], 0)

FOLDS = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED).split(np.zeros(len(y)), y))
CV3 = list(StratifiedKFold(3, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
lr0 = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
lr = lambda: GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
                          {'logisticregression__C': [0.01, 0.1, 1]}, cv=3, scoring='f1_macro', n_jobs=8)
svc = lambda: GridSearchCV(make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True, random_state=SEED)),
                           {'svc__C': [1, 10]}, cv=3, scoring='f1_macro', n_jobs=8)


def run(make, X, folds, per_rep):
    fs, oof = [], {}
    for i, (a, b) in enumerate(folds):
        p = make().fit(X[a], y[a]).predict(X[b]); fs.append(mf1(y[b], p))
        oof.setdefault(i // per_rep, np.zeros(len(y), int))[b] = p
    return np.array(fs), np.mean([mf1(y, o) for o in oof.values()])


res = {}
for name, make, X, folds, per, key in [('B0', lr0, CS, FOLDS, 5, 'B0 color+LR'), ('B1', lr, F, FOLDS, 5, 'B1 feat+LR'),
                                        ('B2', svc, F, FOLDS, 5, 'B2 feat+SVC'), ('S2', lr, D, FOLDS, 5, 'S2 B1+D8'),
                                        ('B1_3fold', lr, F, CV3, 3, 'B1 (3-fold)')]:
    fs, pooled = run(make, X, folds, per)
    assert abs(pooled - rec.loc[key, 'mean']) < 0.0005, (name, pooled, rec.loc[key, 'mean'])
    res[name] = fs
    print(f'{name}: pooled {pooled:.4f} (recorded {rec.loc[key, "mean"]})', flush=True)

nb = json.load(open('problem.ipynb'))
txt = '\n'.join(''.join(o.get('text', '')) for c in nb['cells'] for o in c.get('outputs', []))
ft = np.array([float(m.group(1)) for m in re.finditer(r'\(12, ([0-9.]+)\)\]', txt)])
assert len(ft) == 3 and abs(np.mean(ft) - np.mean(res['B1_3fold']) - rec.loc['FT resnet18@64', 'delta']) < 0.001
res['FT'] = ft  # rounded to 4 decimals in the notebook output

rows = []
for a, b in [('B0', 'B1'), ('B1', 'B2'), ('B1', 'S2'), ('B1_3fold', 'FT')]:
    d = res[b] - res[a]; sd = d.std(ddof=1)
    rows.append(dict(baseline=a, candidate=b, n=len(d), mean=d.mean(), sd=sd, mean_over_sd=d.mean() / sd,
                     better=int((d > 0).sum()), worse=int((d < 0).sum()), diffs=np.round(d, 4).tolist()))
out = pd.DataFrame(rows)
out.to_csv('paired_recheck.csv', index=False)
print(out.drop(columns='diffs').round(4).to_string(index=False))
print(out[['baseline', 'candidate', 'diffs']].to_string(index=False))
