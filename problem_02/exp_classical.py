"""Step 1: classical models, shared 3-fold CV. Usage: python3 exp_classical.py [name ...]"""
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "2"  # 3 fold-workers x 2 threads
import sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from joblib import Parallel, delayed
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from common import data, folds, mf1, save_json, load_json, log, RESULTS


def gamma(Xtr, mult=1.0):  # sklearn's gamma="scale" computed on the fold's training part
    return mult / (Xtr.shape[1] * Xtr.var())


# name -> (feature set, factory(Xtr) -> estimator)
MODELS = {
    "LR": ("pix", lambda X: LogisticRegression(C=1, max_iter=500)),
    "kNN5": ("pix", lambda X: KNeighborsClassifier(5, weights="distance")),
    "SVC_g0.5": ("pix", lambda X: SVC(C=10, gamma=gamma(X, 0.5))),
    "SVC_g1": ("pix", lambda X: SVC(C=10, gamma=gamma(X, 1.0))),
    "SVC_g2": ("pix", lambda X: SVC(C=10, gamma=gamma(X, 2.0))),
    "SVC_C3": ("pix", lambda X: SVC(C=3, gamma=gamma(X, 1.0))),
    "SVC_C30": ("pix", lambda X: SVC(C=30, gamma=gamma(X, 1.0))),
    "RF": ("pix", lambda X: RandomForestClassifier(300, n_jobs=2, random_state=0)),
    "HGB": ("pix", lambda X: HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=0)),
    "MLP": ("pix", lambda X: MLPClassifier((512, 256), early_stopping=True, max_iter=100, random_state=0)),
    "HOG_SVC": ("hog", lambda X: SVC(C=10, gamma=gamma(X))),
    "PCA100_SVC": ("pix", lambda X: make_pipeline(PCA(100, random_state=0), SVC(C=10, gamma=gamma(X)))),
}
# PCA100_SVC: gamma is computed on raw pixels (not PCA space); PCA keeps most variance so this is close to "scale"


def features(X):
    flat = X.reshape(len(X), -1)
    # HOG has no fitted parameters, so computing it once on all images does not leak.
    H = np.stack([hog(im, orientations=9, pixels_per_cell=(7, 7), cells_per_block=(2, 2)) for im in X])
    return {"pix": flat, "hog": H}


def run_fold(F, y, tr, va, feat, factory):
    Xtr, Xva = F[feat][tr], F[feat][va]
    m = factory(Xtr).fit(Xtr, y[tr])
    proba = m.predict_proba(Xva) if hasattr(m, "predict_proba") else None
    return m.predict(Xva), proba


def main(names):
    X, y = data()
    F = features(X)
    fl = folds(y)
    res = load_json("classical.json")
    for name in names:
        feat, factory = MODELS[name]
        t = time.time()
        out = Parallel(n_jobs=3)(delayed(run_fold)(F, y, tr, va, feat, factory) for tr, va in fl)
        sec = time.time() - t
        scores = [mf1(y[va], p) for (tr, va), (p, _) in zip(fl, out)]
        pred = np.zeros(len(y), dtype=int)
        for (tr, va), (p, _) in zip(fl, out):
            pred[va] = p
        np.save(RESULTS / f"oof_pred_{name}.npy", pred)
        if out[0][1] is not None:
            proba = np.zeros((len(y), 7))
            for (tr, va), (_, pr) in zip(fl, out):
                proba[va] = pr
            np.save(RESULTS / f"oof_proba_{name}.npy", proba)
        res[name] = {"fold_f1": scores, "mean": float(np.mean(scores)), "std": float(np.std(scores)), "seconds": sec}
        save_json("classical.json", res)
        log(f"{name}: mean {np.mean(scores):.4f} std {np.std(scores):.4f} folds {np.round(scores, 4).tolist()} {sec:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1:] or list(MODELS))
