# 2026-10-06 사후 검증 2라운드 (EXPERIMENTS.md E10). 실행: PYTHONNOUSERSITE=1 PYTHONPATH=<tsfresh·sktime 격리 설치 폴더> python3 exp_literature_r2.py
# 사전 고정: 기준 = base+S3 + ET + 클래스1 x1.4 / 짝 = StratifiedKFold(5, shuffle, rs=0..9) 50개
# 후보 순서(비용 우선순위): C8 합성크기, C4 2단계, C6 catch22, C5 tsfresh, C7 MiniRocket
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

DATA = "/home/akswosla/maicon/problem_04/"
cols = ["force_x", "force_y", "force_z", "torque_x", "torque_y", "torque_z"]
tr = pd.read_csv(DATA + "train_series.csv"); lb = pd.read_csv(DATA + "train_labels.csv")
g = {k: v.sort_values("step")[cols].to_numpy(float) for k, v in tr.groupby("sample_id")}
X = np.stack([g[k] for k in lb.sample_id]); y = lb.label.to_numpy()
W = np.array([1, 1.4, 1, 1, 1])

def stats(s): return np.hstack([s.mean(1), s.std(1), s.min(1), s.max(1), s[:, -1] - s[:, 0], np.abs(np.diff(s, axis=1)).mean(1)])
def s3(s):
    A = lambda m, n: s[:, m - 1:n].mean(1)
    avg = [A(1, 3), A(4, 6), A(7, 9), A(10, 12), A(13, 15), A(1, 5), A(6, 10), A(11, 15), A(1, 15)]
    D = lambda m, n: (s[:, n - 1] - s[:, m - 1]) / (n - m)
    trend = np.where(A(1, 5) < A(11, 15), 1.0, -1.0)
    M = ((s[:, :-1] * trend[:, None, :]) < (s[:, 1:] * trend[:, None, :])).sum(1)
    return np.hstack(avg + [D(1, 8), D(8, 15), M])
def base_feats(s): return np.hstack([stats(s), s3(s)])

# C8: 합성 크기 프로파일 8개
fx, fy, fz, tx, ty, tz = [X[:, :, i] for i in range(6)]
mag = np.stack([np.hypot(fx, fy), np.hypot(fx, fz), np.hypot(fy, fz), np.sqrt(fx**2 + fy**2 + fz**2),
                np.hypot(tx, ty), np.hypot(tx, tz), np.hypot(ty, tz), np.sqrt(tx**2 + ty**2 + tz**2)], axis=2)
BASE = base_feats(X)
F_C8 = np.hstack([BASE, base_feats(mag)])

# C6: catch22 (샘플별 비지도 추출 → 누출 없음)
from sktime.transformations.panel.catch22 import Catch22
X3 = X.transpose(0, 2, 1)  # (n, channels, time)
c22 = np.nan_to_num(np.asarray(Catch22().fit_transform(X3), dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
F_C6 = np.hstack([BASE, c22])

# C5: tsfresh Efficient (필터 없음, NaN/inf -> 0)
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
long = tr.rename(columns={"sample_id": "id", "step": "time"})
ts = extract_features(long, column_id="id", column_sort="time", default_fc_parameters=EfficientFCParameters(),
                      disable_progressbar=True, n_jobs=4).loc[lb.sample_id]
ts = np.nan_to_num(ts.to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
F_C5 = np.hstack([BASE, ts])
print("feature counts: base", BASE.shape[1], "C8", F_C8.shape[1], "C6", F_C6.shape[1], "C5", F_C5.shape[1], flush=True)

from sktime.transformations.panel.rocket import MiniRocketMultivariate
def ET(): return ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1)
def weighted_pred(m, f): return m.classes_[(m.predict_proba(f) * W).argmax(1)]
mf = lambda t, p: float(f1_score(t, p, labels=range(5), average="macro", zero_division=0))

names = ["base", "C8 magnitude", "C4 two-stage", "C6 catch22", "C5 tsfresh", "C7 MiniRocket"]
scores = {k: [] for k in names}; oof = {k: [] for k in names}
for rep in range(10):
    P = {k: np.zeros(len(y), int) for k in names}
    for a, b in StratifiedKFold(5, shuffle=True, random_state=rep).split(BASE, y):
        m = ET().fit(BASE[a], y[a]); pb = weighted_pred(m, BASE[b]); P["base"][b] = pb
        P["C8 magnitude"][b] = weighted_pred(ET().fit(F_C8[a], y[a]), F_C8[b])
        # C4: 기준 예측이 1/2인 샘플만 1-vs-2 전용 이진 ET로 재분류
        a12 = a[np.isin(y[a], [1, 2])]
        m12 = ET().fit(BASE[a12], y[a12])
        p4 = pb.copy(); sel = np.isin(pb, [1, 2])
        if sel.any(): p4[sel] = m12.predict(BASE[b][sel])
        P["C4 two-stage"][b] = p4
        P["C6 catch22"][b] = weighted_pred(ET().fit(F_C6[a], y[a]), F_C6[b])
        P["C5 tsfresh"][b] = weighted_pred(ET().fit(F_C5[a], y[a]), F_C5[b])
        mr = MiniRocketMultivariate(random_state=42).fit(X3[a])
        clf = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)))
        clf.fit(mr.transform(X3[a]), y[a]); P["C7 MiniRocket"][b] = clf.predict(mr.transform(X3[b]))
        for k in names: scores[k].append(mf(y[b], P[k][b]))
    for k in names: oof[k].append(P[k])
    print("rep", rep, "done", flush=True)

b = np.array(scores["base"], dtype=np.float64)
print(f"\npairs={len(b)}  base mean fold F1={b.mean():.4f}")
for k in names[1:]:
    d = np.array(scores[k], dtype=np.float64) - b
    win = int((d > 1e-9).sum()); loss = int((d < -1e-9).sum()); tie = len(d) - win - loss
    print(f"{k:14s} mean d={d.mean():+.4f}  sd={d.std():.4f}  win/loss/tie={win}/{loss}/{tie}  -> {'PASS' if d.mean() > 0 and win > loss else 'fail'}")
print()
for k in names:
    c12 = np.mean([((y == 1) & (o == 2)).sum() + ((y == 2) & (o == 1)).sum() for o in oof[k]])
    f = np.mean([f1_score(y, o, labels=range(5), average=None, zero_division=0) for o in oof[k]], 0)
    print(f"{k:14s} 1<->2 confusions/run={c12:.1f}  per-class F1={np.round(f, 2)}")
