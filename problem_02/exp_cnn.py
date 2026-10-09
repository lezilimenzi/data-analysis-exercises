"""Step 2: CNN screening, shared 3-fold CV.
Usage: python3 exp_cnn.py C0 C1 ...   |   python3 exp_cnn.py --summary
"""
import sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from common import data, folds, mf1, save_json, load_json, log, RESULTS

torch.set_num_threads(8)
EPOCHS, BATCH = 20, 128
# aug flags: shift, flip, erase
CONFIGS = {
    "C0": dict(shift=False, flip=False, erase=False),
    "C1": dict(shift=True, flip=False, erase=False),
    "C2": dict(shift=True, flip=True, erase=False),
    "C3": dict(shift=True, flip=False, erase=True),
    "LS": dict(shift=False, flip=False, erase=False, ls=0.1),  # Step 6a: C0 + label smoothing
}


def block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))


def make_model(w=32):
    return nn.Sequential(block(1, w), block(w, 2 * w), block(2 * w, 4 * w),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(4 * w, 7))


def shift(x, k=2):  # per-sample random translation, empty area = 0
    B = len(x)
    p = F.pad(x, (k, k, k, k))
    dx, dy = torch.randint(0, 2 * k + 1, (2, B))
    return torch.stack([p[i, :, dy[i]:dy[i] + 28, dx[i]:dx[i] + 28] for i in range(B)])


def erase(x):  # with prob 0.5 zero a random 4~10 px square
    x = x.clone()
    for i in range(len(x)):
        if torch.rand(1).item() < 0.5:
            s = int(torch.randint(4, 11, (1,)))
            r, c = int(torch.randint(0, 29 - s, (1,))), int(torch.randint(0, 29 - s, (1,)))
            x[i, :, r:r + s, c:c + s] = 0
    return x


def flip(x):
    return torch.where(torch.rand(len(x), 1, 1, 1) < 0.5, x.flip(3), x)


def train_predict(Xtr, ytr, Xev, aug, seed, epochs=EPOCHS, width=32, lr=2e-3, on_epoch=None):
    """Train on (Xtr, ytr) only; return softmax probs for Xev (numpy). X: (N,28,28) float 0~1.
    on_epoch(ep, model): optional diagnostic hook called after each epoch (model left in eval mode)."""
    torch.manual_seed(seed)
    Xt = torch.tensor(Xtr).unsqueeze(1)
    yt = torch.tensor(ytr)
    m = make_model(width)
    steps = epochs * ((len(Xt) + BATCH - 1) // BATCH)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    t0 = time.time()
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BATCH):
            b = perm[i:i + BATCH]
            x = Xt[b]
            if aug["shift"]: x = shift(x)
            if aug["flip"]: x = flip(x)
            if aug["erase"]: x = erase(x)
            opt.zero_grad()
            F.cross_entropy(m(x), yt[b], label_smoothing=aug.get("ls", 0.0)).backward()
            opt.step()
            sched.step()
        if ep == 0:
            log(f"    epoch time {time.time() - t0:.1f}s")
        if on_epoch is not None:
            m.eval()
            with torch.no_grad():
                on_epoch(ep, m)
    m.eval()
    with torch.no_grad():
        return F.softmax(m(torch.tensor(Xev).unsqueeze(1)), 1).numpy()


def run(name, seed_base=0):
    X, y = data()
    fl = folds(y)
    t = time.time()
    proba = np.zeros((len(y), 7))
    scores = []
    for k, (tr, va) in enumerate(fl):
        proba[va] = train_predict(X[tr], y[tr], X[va], CONFIGS[name], seed=k + 10 * seed_base)
        scores.append(mf1(y[va], proba[va].argmax(1)))
        log(f"  {name} fold{k}: {scores[-1]:.4f}")
    sec = time.time() - t
    tag = name if seed_base == 0 else f"{name}_s{seed_base}"
    np.save(RESULTS / f"oof_proba_cnn_{tag}.npy", proba)
    res = load_json("cnn.json" if seed_base == 0 else "cnn_seeds.json")
    res[tag] = {"aug": CONFIGS[name], "epochs": EPOCHS, "fold_f1": scores, "mean": float(np.mean(scores)),
                 "std": float(np.std(scores)), "seconds": sec}
    save_json("cnn.json" if seed_base == 0 else "cnn_seeds.json", res)
    log(f"{tag}: mean {np.mean(scores):.4f} std {np.std(scores):.4f} {sec:.0f}s")


def summary():
    """Adoption rule vs C0: paired fold-diff mean >= +0.003 and positive in >=2 of 3 folds."""
    res = load_json("cnn.json")
    base = np.array(res["C0"]["fold_f1"])
    best, best_mean = "C0", res["C0"]["mean"]
    for n, r in res.items():
        d = np.array(r["fold_f1"]) - base
        ok = n != "C0" and d.mean() >= 0.003 and (d > 0).sum() >= 2
        print(f"{n}: mean {r['mean']:.4f} diff_vs_C0 {np.round(d, 4).tolist()} mean_diff {d.mean():+.4f} adopt={ok}")
        if ok and r["mean"] > best_mean:
            best, best_mean = n, r["mean"]
    print("selected:", best)


if __name__ == "__main__":
    if sys.argv[1:] == ["--summary"]:
        summary()
    else:
        for n in sys.argv[1:]:  # "C0" or "C0:1" (config:seed_base, extra seeds for the ensemble check)
            run(*(n.split(":")[0], int(n.split(":")[1])) if ":" in n else (n,))
