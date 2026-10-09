"""Step 3: equal-weight ensembles from 3-fold OOF probabilities. Usage: python3 exp_ensemble.py"""
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "2"
import time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from joblib import Parallel, delayed
from sklearn.svm import SVC
from common import data, folds, mf1, save_json, log, RESULTS


def svc_fold(X, y, tr, va):
    g = 1.0 / (X.shape[1] * X[tr].var())
    return SVC(C=10, gamma=g, probability=True, random_state=0).fit(X[tr], y[tr]).predict_proba(X[va])


def main():
    X, y = data()
    fl = folds(y)
    flat = X.reshape(len(X), -1)
    p = RESULTS / "oof_proba_SVC_prob.npy"
    if not p.exists():
        t = time.time()
        out = Parallel(n_jobs=3)(delayed(svc_fold)(flat, y, tr, va) for tr, va in fl)
        proba = np.zeros((len(y), 7))
        for (tr, va), pr in zip(fl, out):
            proba[va] = pr
        np.save(p, proba)
        log(f"SVC probability OOF: {time.time() - t:.0f}s")
    P = {"CNN": np.load(RESULTS / "oof_proba_cnn_C0.npy"), "SVC": np.load(p),
         "MLP": np.load(RESULTS / "oof_proba_MLP.npy"), "HGB": np.load(RESULTS / "oof_proba_HGB.npy")}
    combos = [["CNN"], ["SVC"], ["CNN", "SVC"], ["CNN", "MLP"], ["CNN", "SVC", "MLP"], ["CNN", "SVC", "MLP", "HGB"]]
    res = {}
    for c in combos:
        avg = np.mean([P[m] for m in c], axis=0).argmax(1)
        f1s = [mf1(y[va], avg[va]) for _, va in fl]
        res["+".join(c)] = {"fold_f1": f1s, "mean": float(np.mean(f1s)), "std": float(np.std(f1s))}
    base = np.array(res["CNN"]["fold_f1"]); thr = res["CNN"]["std"]
    for k, r in res.items():
        d = np.array(r["fold_f1"]) - base
        r["adopt_vs_CNN"] = bool(k != "CNN" and d.mean() >= thr)
        print(f"{k:22s} mean {r['mean']:.4f} std {r['std']:.4f} diff_vs_CNN {np.round(d, 4).tolist()} mean_diff {d.mean():+.4f} adopt={r['adopt_vs_CNN']}")
    save_json("ensemble.json", res)


if __name__ == "__main__":
    main()
