"""Post-hoc check (2026-09-28): class 1-vs-2 discrimination INSIDE the original 5-class model
(stats36 + ExtraTrees, same folds as the notebook's weight check: StratifiedKFold(5, shuffle, random_state=rep), rep 0..9).
Score = p2 / (p1 + p2) on true class-1/2 samples. Train data only. Output: check_pair_auc_5class.out
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from exp_order import load, stats36, oof_proba

X, y, ids = load()
S = stats36(X)
m = np.isin(y, [1, 2])
y12 = (y[m] == 2).astype(int)
aucs, exact = [], []
for r in range(10):
    P = oof_proba(S, y, r, 5)
    aucs.append(roc_auc_score(y12, P[m, 2] / (P[m, 1] + P[m, 2] + 1e-12)))
    exact.append((P[m].argmax(1) == y[m]).mean())
print(f"5-class model, classes 1 vs 2: AUC of p2/(p1+p2) = {np.mean(aucs):.3f} +/- {np.std(aucs):.3f} (10 reps)")
print(f"per-rep AUC: {np.round(aucs, 3).tolist()}")
print(f"share of true 1/2 samples whose argmax is exactly right: {np.mean(exact):.3f}")
