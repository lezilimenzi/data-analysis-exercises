"""Post-hoc check (2026-10-02): per-repetition paired gain of the class-1 decision weight used in the notebook.
Same model (stats36 + ExtraTrees) and folds as the notebook's weight check (StratifiedKFold(5, shuffle, random_state=rep), rep 0..9).
This is the same OOF the weight was chosen on, so it is optimistic. Output: check_weight_paired.out
"""
import numpy as np
from exp_order import load, stats36, oof_proba, macro_f1

X, y, ids = load()
S = stats36(X)
d = []
for r in range(10):
    P = oof_proba(S, y, r, 5)
    d.append(macro_f1(y, (P * [1, 1.4, 1, 1, 1]).argmax(1)) - macro_f1(y, P.argmax(1)))
d = np.array(d)
print("per-rep diff (weighted - unweighted):", np.round(d, 3).tolist())
print(f"mean {d.mean():+.3f}, sd {d.std():.3f}, positive {int((d > 0).sum())}/10")
