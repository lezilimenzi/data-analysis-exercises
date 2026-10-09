# 2026-10-05 사후 검증 1라운드 (EXPERIMENTS.md E9). train만 사용.
# 사전 고정: 기준 = stats36 + ET + 클래스1 x1.4 / 짝 = StratifiedKFold(5, shuffle, rs=0..9) 50개
# 채택: 평균 d > 0 이고 개선 > 악화 / 후보 순서: C1 FFT, C2 S3, C3 FFT+S3
# 추가: C2와 기준의 예측이 다른 OOF 샘플에서 C2가 고친/망친 비율 (채점 2 해석용)
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

cols = ["force_x", "force_y", "force_z", "torque_x", "torque_y", "torque_z"]
tr = pd.read_csv("train_series.csv"); lb = pd.read_csv("train_labels.csv")
g = {k: v.sort_values("step")[cols].to_numpy(float) for k, v in tr.groupby("sample_id")}
X = np.stack([g[k] for k in lb.sample_id]); y = lb.label.to_numpy()

def stats(s): return np.hstack([s.mean(1), s.std(1), s.min(1), s.max(1), s[:, -1] - s[:, 0], np.abs(np.diff(s, axis=1)).mean(1)])
def fft_feats(s):  # 원 논문 S4: 마지막 값을 반복해 N=16, 고조파 1..8 진폭
    p = np.concatenate([s, s[:, -1:, :]], axis=1)
    return (np.abs(np.fft.fft(p, axis=1)) / 16)[:, 1:9, :].reshape(len(s), -1)
def s3_feats(s):   # 원 논문 S3: 다중 해상도 평균 9 + D(1,8), D(8,15) + 단조성 = 채널당 12
    A = lambda m, n: s[:, m - 1:n].mean(1)
    avg = [A(1, 3), A(4, 6), A(7, 9), A(10, 12), A(13, 15), A(1, 5), A(6, 10), A(11, 15), A(1, 15)]
    D = lambda m, n: (s[:, n - 1] - s[:, m - 1]) / (n - m)
    trend = np.where(A(1, 5) < A(11, 15), 1.0, -1.0)
    M = ((s[:, :-1] * trend[:, None, :]) < (s[:, 1:] * trend[:, None, :])).sum(1)
    return np.hstack(avg + [D(1, 8), D(8, 15), M])

base = stats(X); fx = fft_feats(X); s3 = s3_feats(X)
F = {"base": base, "C1 base+FFT": np.hstack([base, fx]), "C2 base+S3": np.hstack([base, s3]), "C3 base+FFT+S3": np.hstack([base, fx, s3])}
print({k: v.shape[1] for k, v in F.items()})
W = np.array([1, 1.4, 1, 1, 1])
scores = {k: [] for k in F}; oof = {k: [] for k in F}
for rep in range(10):
    P = {k: np.zeros(len(y), int) for k in F}
    for a, b in StratifiedKFold(5, shuffle=True, random_state=rep).split(base, y):
        for k, f in F.items():
            m = ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1).fit(f[a], y[a])
            p = m.classes_[(m.predict_proba(f[b]) * W).argmax(1)]
            scores[k].append(float(f1_score(y[b], p, labels=range(5), average="macro", zero_division=0))); P[k][b] = p
    for k in F: oof[k].append(P[k])
b = np.array(scores["base"], dtype=np.float64)
print(f"pairs={len(b)}  base mean fold F1={b.mean():.4f}")
for k in list(F)[1:]:
    d = np.array(scores[k], dtype=np.float64) - b
    win = int((d > 1e-9).sum()); loss = int((d < -1e-9).sum()); tie = len(d) - win - loss
    print(f"{k:16s} mean d={d.mean():+.4f}  sd={d.std():.4f}  win/loss/tie={win}/{loss}/{tie}  -> {'PASS' if d.mean() > 0 and win > loss else 'fail'}")
for k in F:
    c12 = np.mean([((y == 1) & (o == 2)).sum() + ((y == 2) & (o == 1)).sum() for o in oof[k]])
    f = np.mean([f1_score(y, o, labels=range(5), average=None, zero_division=0) for o in oof[k]], 0)
    acc = np.mean([(o == y).mean() for o in oof[k]]); mf = [f1_score(y, o, labels=range(5), average="macro", zero_division=0) for o in oof[k]]
    print(f"{k:16s} 1<->2 confusions/run={c12:.1f}  per-class F1={np.round(f, 2)}  OOF acc={acc:.3f}  OOF macroF1={np.mean(mf):.3f} ({min(mf):.3f}-{max(mf):.3f})")
fix = brk = both = 0
for ob, oc in zip(oof["base"], oof["C2 base+S3"]):
    d = ob != oc; fix += int(((oc == y) & d).sum()); brk += int(((ob == y) & d).sum()); both += int(((ob != y) & (oc != y) & d).sum())
n = fix + brk + both
print(f"C2 vs base changed OOF preds: {n} over 10 runs; fixes {fix/n:.0%}, breaks {brk/n:.0%}, both wrong {both/n:.0%}; P(6 of 6 fixes) ~ {(fix/n)**6:.3f}")
