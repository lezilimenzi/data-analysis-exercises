"""Step 5: error analysis from saved 3-fold OOF predictions (no training, test not read)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import data, RESULTS

X, y = data()
P = [np.load(RESULTS / f"oof_proba_cnn_{t}.npy") for t in ("C0", "C0_s1", "C0_s2")]
ens = np.mean(P, 0)
pred = ens.argmax(1)
svc = np.load(RESULTS / "oof_pred_SVC_g1.npy")
seed_wrong = np.stack([p.argmax(1) != y for p in P])  # (3, N)

err = pred != y
print("total errors", err.sum())
for a, b, name in [(2, 6, "coat->pullover"), (6, 2, "pullover->coat")]:
    m = (y == a) & (pred == b)
    idx = np.where(m)[0]
    conf = ens[idx].max(1)
    second = (np.argsort(-ens[idx], 1)[:, 1] == a).mean() if len(idx) else 0
    print(f"{name}: n={len(idx)} mean_conf={conf.mean():.3f} median_conf={np.median(conf):.3f} "
          f"true_is_2nd={second:.2f} all3seeds_wrong={seed_wrong[:, idx].all(0).mean():.2f} "
          f"any_seed_wrong={seed_wrong[:, idx].any(0).mean():.2f} svc_also_wrong={(svc[idx] != a).mean():.2f}")
allerr = np.where(err)[0]
print("ALL errors: mean_conf %.3f, all3seeds_wrong %.2f, any_seed_wrong(of ens errors) %.2f, svc_also_wrong %.2f" % (
    ens[allerr].max(1).mean(), seed_wrong[:, allerr].all(0).mean(), seed_wrong[:, allerr].any(0).mean(),
    (svc[allerr] != y[allerr]).mean()))
print("mean conf on correct: %.3f" % ens[~err].max(1).mean())
# how many images are wrong for any single seed but right for the ensemble / vice versa
print("images wrong for >=1 seed:", seed_wrong.any(0).sum(), " all 3 seeds wrong:", seed_wrong.all(0).sum(), " ens wrong:", err.sum())

# montage of coat<->pullover errors
sel = np.where(err & np.isin(y, (2, 6)) & np.isin(pred, (2, 6)))[0]
sel = sel[np.argsort(y[sel], kind="stable")]
cols = 17
rows = int(np.ceil(len(sel) / cols))
fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.05, rows * 1.25))
for ax in axes.ravel():
    ax.axis("off")
for ax, i in zip(axes.ravel(), sel):
    ax.imshow(X[i], cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"{'C' if y[i] == 2 else 'P'}>{'C' if pred[i] == 2 else 'P'} {ens[i].max():.2f}", fontsize=7)
fig.suptitle("coat/pullover errors: true>pred (C=coat, P=pullover) and confidence", fontsize=10)
plt.tight_layout()
plt.savefig(RESULTS / "errors_coat_pullover.png", dpi=110)
print("saved montage of", len(sel))
