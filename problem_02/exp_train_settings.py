"""PLAN §14: post-hoc check of CNN training settings, same 3-fold, same seeds as saved C0 (seed = fold + 10*s).
Usage: python3 exp_train_settings.py E0 E1 E2a E2b E3   |   python3 exp_train_settings.py --summary
Finished seeds are saved immediately and skipped on re-run. test data is never read.
"""
import sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import wilcoxon
from common import data, folds, mf1, save_json, load_json, log, RESULTS
from exp_cnn import CONFIGS, train_predict
import torch
import torch.nn.functional as F

C0, SHIFT = CONFIGS["C0"], CONFIGS["C1"]
SETTINGS = {  # name: (aug, epochs, lr, seed_bases, baseline)
    "E0": (C0, 20, 2e-3, [0], "C0"),           # reproducibility check of saved C0 s0
    "E1": (C0, 40, 2e-3, [0, 1, 2], "C0"),
    "E2a": (C0, 20, 1e-3, [0, 1, 2], "C0"),
    "E2b": (C0, 20, 4e-3, [0, 1, 2], "C0"),
    "E3": (SHIFT, 40, 2e-3, [0, 1, 2], "E1"),  # only if E1 is adopted
}
OUT = "train_settings.json"


def c0_baseline():
    """Saved C0 results: fold F1 per seed base and OOF probas (seed bases 0, 1, 2)."""
    a, b = load_json("cnn.json"), load_json("cnn_seeds.json")
    fold = {0: a["C0"]["fold_f1"], 1: b["C0_s1"]["fold_f1"], 2: b["C0_s2"]["fold_f1"]}
    proba = {s: np.load(RESULTS / f"oof_proba_cnn_{t}.npy") for s, t in [(0, "C0"), (1, "C0_s1"), (2, "C0_s2")]}
    return fold, proba


def load_setting(name):
    r = load_json(OUT).get(name, {})
    fold = {int(s): v["fold_f1"] for s, v in r.get("seeds", {}).items()}
    proba = {s: np.load(RESULTS / f"oof_proba_ts_{name}_s{s}.npy") for s in fold}
    return fold, proba


def run(name):
    aug, epochs, lr, seed_bases, _ = SETTINGS[name]
    X, y = data()
    fl = folds(y)
    res = load_json(OUT)
    entry = res.setdefault(name, {"aug": aug, "epochs": epochs, "lr": lr, "seeds": {}})
    for s in seed_bases:
        if str(s) in entry["seeds"]:
            log(f"{name} s{s}: already done, skipped")
            continue
        t = time.time()
        proba = np.zeros((len(y), 7))
        scores, curves = [], []
        for k, (tr, va) in enumerate(fl):
            curve = []
            Xva = torch.tensor(X[va]).unsqueeze(1)
            hook = lambda ep, m: curve.append(mf1(y[va], F.softmax(m(Xva), 1).argmax(1).numpy()))
            proba[va] = train_predict(X[tr], y[tr], X[va], aug, seed=k + 10 * s, epochs=epochs, lr=lr, on_epoch=hook)
            scores.append(mf1(y[va], proba[va].argmax(1)))
            curves.append(curve)
            log(f"  {name} s{s} fold{k}: {scores[-1]:.4f}")
        np.save(RESULTS / f"oof_proba_ts_{name}_s{s}.npy", proba)
        entry["seeds"][str(s)] = {"fold_f1": scores, "curves": curves, "seconds": time.time() - t}
        save_json(OUT, res)
        log(f"{name} s{s}: mean {np.mean(scores):.4f} {time.time() - t:.0f}s")


def judge(name):
    """PLAN §14.3: vs baseline, 9 paired (fold x seed) diffs: mean >= +0.003, >= 6 positive,
    3-seed ensemble OOF Macro-F1 not below baseline ensemble."""
    y = data()[1]
    base_name = SETTINGS[name][4]
    bf, bp = c0_baseline() if base_name == "C0" else load_setting(base_name)
    nf, npb = load_setting(name)
    seeds = sorted(set(bf) & set(nf))
    d = np.array([np.array(nf[s]) - np.array(bf[s]) for s in seeds]).ravel()
    ens_b = mf1(y, sum(bp[s] for s in seeds).argmax(1))
    ens_n = mf1(y, sum(npb[s] for s in seeds).argmax(1))
    ok = len(seeds) == 3 and d.mean() >= 0.003 and (d > 0).sum() >= 6 and ens_n >= ens_b
    p = wilcoxon(d).pvalue if len(d) >= 6 and np.any(d != 0) else float("nan")
    return {"baseline": base_name, "seeds": seeds, "diffs": d.round(4).tolist(), "mean_diff": float(d.mean()),
            "n_pos": int((d > 0).sum()), "ens_base": float(ens_b), "ens_new": float(ens_n),
            "wilcoxon_p": float(p), "adopt": bool(ok)}


def summary():
    res = load_json(OUT)
    bf, _ = c0_baseline()
    if "E0" in res:
        diff = np.array(res["E0"]["seeds"]["0"]["fold_f1"]) - np.array(bf[0])
        print(f"E0 reproducibility vs saved C0 s0: diffs {diff.round(4).tolist()} max|diff| {np.abs(diff).max():.4f}")
    for name in ["E1", "E2a", "E2b", "E3"]:
        if name in res and len(res[name]["seeds"]) == 3:
            j = judge(name)
            res[name]["judge"] = j
            print(f"{name} vs {j['baseline']}: mean_diff {j['mean_diff']:+.4f} pos {j['n_pos']}/9 "
                  f"ens {j['ens_base']:.4f}->{j['ens_new']:.4f} wilcoxon_p {j['wilcoxon_p']:.3f} adopt={j['adopt']}")
    save_json(OUT, res)


if __name__ == "__main__":
    if sys.argv[1:] == ["--summary"]:
        summary()
    else:
        for n in sys.argv[1:]:
            if n == "E3" and not (len(load_json(OUT).get("E1", {}).get("seeds", {})) == 3 and judge("E1")["adopt"]):
                log("E3 skipped: E1 not adopted (PLAN §14.2)")
                continue
            run(n)
        summary()
