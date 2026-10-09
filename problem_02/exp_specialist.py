"""Step 6b: coat/pullover specialist re-classification, same 3-fold, baseline = C0 seed 0 OOF."""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from common import data, folds, mf1, save_json, log, RESULTS
from exp_cnn import CONFIGS, train_predict
from sklearn.metrics import f1_score

X, y = data()
base = np.load(RESULTS / "oof_proba_cnn_C0.npy")
pred = base.argmax(1)
new = pred.copy()
for k, (tr, va) in enumerate(folds(y)):
    sub = tr[np.isin(y[tr], (2, 6))]
    ysub = (y[sub] == 6).astype(np.int64)  # coat -> 0, pullover -> 1 (uses first 2 of 7 logits)
    tgt = va[np.isin(pred[va], (2, 6))]  # only samples the generalist sent to coat/pullover
    p = train_predict(X[sub], ysub, X[tgt], CONFIGS["C0"], seed=k)[:, :2]
    new[tgt] = np.where(p.argmax(1) == 1, 6, 2)
    log(f"  spec fold{k}: train {len(sub)} reclassified {len(tgt)} changed {(new[tgt] != pred[tgt]).sum()}")
fl = folds(y)
b = [mf1(y[va], pred[va]) for _, va in fl]
n = [mf1(y[va], new[va]) for _, va in fl]
d = np.array(n) - np.array(b)
f1b, f1n = f1_score(y, pred, average=None), f1_score(y, new, average=None)
res = {"baseline_fold_f1": b, "specialist_fold_f1": n, "mean_base": float(np.mean(b)), "mean_spec": float(np.mean(n)),
       "diff": d.tolist(), "f1_coat_base_new": [float(f1b[2]), float(f1n[2])], "f1_pullover_base_new": [float(f1b[6]), float(f1n[6])]}
save_json("specialist.json", res)
log(f"specialist: base {np.mean(b):.4f} new {np.mean(n):.4f} diff {np.round(d, 4).tolist()} mean_diff {d.mean():+.4f} "
    f"coatF1 {f1b[2]:.3f}->{f1n[2]:.3f} pullF1 {f1b[6]:.3f}->{f1n[6]:.3f} adopt={d.mean() >= 0.003 and (d > 0).sum() >= 2}")
