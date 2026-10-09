"""plan.md '사후 검증 계획: 1번·2번 혼동'. Train data only (test is never read). Existing files untouched.
Stage 1: order/timing features (order30) for class 1 vs 2 (52 samples): (a) permutation pairs, (b) univariate
AUC vs label-shuffle null, (c) multivariate ET AUC, (d) label-reliability count (reference).
Stage 2 (only if stage 1 passes): 5-class stats36 vs stats36+order30, class-1 weight chosen in nested CV.
Output: exp_order_results.json
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

CHANNELS = ["force_x", "force_y", "force_z", "torque_x", "torque_y", "torque_z"]
LABELS = [0, 1, 2, 3, 4]
WEIGHTS = (1.0, 1.2, 1.4, 1.6, 2.0)
PAIRS = [("SEG-1057", "SEG-1014"), ("SEG-1061", "SEG-1100")]  # Codex permutation_groups.csv (label 1 / label 2)


def load():
    s = pd.read_csv("train_series.csv", encoding="utf-8-sig")
    lab = pd.read_csv("train_labels.csv", encoding="utf-8-sig")
    g = {k: v.sort_values("step") for k, v in s.groupby("sample_id")}
    X = np.stack([g[i][CHANNELS].to_numpy(float) for i in lab.sample_id])  # (120, 15, 6), same as notebook
    assert X.shape == (120, 15, 6) and np.isfinite(X).all()
    return X, lab.label.to_numpy(), lab.sample_id.to_numpy()


def stats36(s):  # identical to problem.ipynb stats_features
    return np.hstack([s.mean(1), s.std(1), s.min(1), s.max(1), s[:, -1] - s[:, 0], np.abs(np.diff(s, axis=1)).mean(1)])


def order30(s):
    t = np.arange(s.shape[1])[None, :, None]
    dev = np.abs(s - s[:, :1])
    den = dev.sum(1)
    centroid = np.divide((t * dev).sum(1), den, out=np.zeros_like(den), where=den > 0)
    return np.hstack([np.abs(s).argmax(1), s.argmax(1), s.argmin(1), centroid,
                      s[:, :7].mean(1) - s[:, 8:].mean(1)]).astype(float)


def model():
    return ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1)


def macro_f1(y, p):
    return f1_score(y, p, labels=LABELS, average="macro", zero_division=0)


def uni_auc(F, y):  # direction-agnostic AUC per feature
    a = np.array([roc_auc_score(y, f) if np.ptp(f) > 0 else 0.5 for f in F.T])
    return np.maximum(a, 1 - a)


def oof_proba(F, y, rep, n_classes):
    P = np.zeros((len(y), n_classes))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=rep).split(F, y):
        P[va] = model().fit(F[tr], y[tr]).predict_proba(F[va])
    return P


def choose_weight(y, P):  # STRATEGY.md 10.3: best inner OOF Macro-F1, ties -> smaller weight
    scores = []
    for w in WEIGHTS:
        Q = P.copy(); Q[:, 1] *= w
        scores.append(macro_f1(y, Q.argmax(1)))
    return WEIGHTS[int(np.argmax(scores))]


def main():
    X, y, ids = load()
    S, O = stats36(X), order30(X)
    res = {}

    # sanity: reproduce the original notebook's weight check (stats36, 5-fold x 10, w=1.0 -> 0.725, 1.4 -> 0.758)
    base = [oof_proba(S, y, r, 5) for r in range(10)]
    f10 = np.mean([macro_f1(y, P.argmax(1)) for P in base])
    f14 = np.mean([macro_f1(y, (P * [1, 1.4, 1, 1, 1]).argmax(1)) for P in base])
    print(f"sanity: stats36 OOF Macro-F1 w=1.0 {f10:.3f} (record 0.725), w=1.4 {f14:.3f} (record 0.758)")
    res["sanity"] = {"w1.0": f10, "w1.4": f14}

    # ---------- stage 1: classes 1 vs 2 only ----------
    m = np.isin(y, [1, 2])
    y12 = (y[m] == 2).astype(int)
    S12, O12, ids12 = S[m], O[m], ids[m]
    idx = {k: i for i, k in enumerate(ids)}

    # (a) permutation pairs: same row multiset, different order; order30 must tell them apart
    a_ok = True
    for p, q in PAIRS:
        same_multiset = sorted(map(tuple, X[idx[p]])) == sorted(map(tuple, X[idx[q]]))
        diff_o = int((O[idx[p]] != O[idx[q]]).sum())
        diff_s = int((~np.isclose(S[idx[p]], S[idx[q]])).sum())
        a_ok &= same_multiset and diff_o > 0
        print(f"(a) {p}(1)/{q}(2): same row multiset {same_multiset}, order30 features differing {diff_o}/30, "
              f"stats36 differing {diff_s}/36")
        res[f"pair_{p}_{q}"] = {"same_multiset": same_multiset, "order30_diff": diff_o, "stats36_diff": diff_s}

    # (b) univariate AUC vs label-shuffle null of the max over features
    rng = np.random.default_rng(0)
    au_o, au_s = uni_auc(O12, y12), uni_auc(S12, y12)
    null = np.array([uni_auc(O12, rng.permutation(y12)).max() for _ in range(200)])
    q95 = float(np.quantile(null, 0.95))
    b_ok = au_o.max() > q95
    print(f"(b) max univariate AUC: order30 {au_o.max():.3f} (feature {int(au_o.argmax())}), stats36 {au_s.max():.3f}; "
          f"shuffled-label null 95th pct of order30 max = {q95:.3f} -> pass {b_ok}")
    res["b"] = {"order30_max": float(au_o.max()), "order30_argmax": int(au_o.argmax()), "stats36_max": float(au_s.max()),
                "null_q95": q95}

    # (c) multivariate 1-vs-2 AUC, same folds, stats36 vs stats36+order30
    SO12 = np.hstack([S12, O12])
    auc_s = np.array([roc_auc_score(y12, oof_proba(S12, y12, r, 2)[:, 1]) for r in range(10)])
    auc_so = np.array([roc_auc_score(y12, oof_proba(SO12, y12, r, 2)[:, 1]) for r in range(10)])
    d = auc_so - auc_s
    c_ok = d.mean() >= 0.03
    print(f"(c) 1-vs-2 OOF AUC (5-fold x 10): stats36 {auc_s.mean():.3f} +/- {auc_s.std():.3f}, "
          f"+order30 {auc_so.mean():.3f} +/- {auc_so.std():.3f}, paired diff {d.mean():+.3f} "
          f"(positive in {(d > 0).sum()}/10) -> pass {c_ok}")
    res["c"] = {"stats36": auc_s.tolist(), "stats36_order30": auc_so.tolist()}

    # (d) label-reliability reference (confident-learning-like, 5-class stats36, OOF proba averaged over 10 reps)
    Pm = np.mean(base, axis=0)
    thr = np.array([Pm[y == k, k].mean() for k in LABELS])
    flag = (Pm[np.arange(len(y)), y] < thr[y]) & (Pm.argmax(1) != y)
    flagged = [str(i) for i in ids[flag & np.isin(y, [1, 2])]]
    print(f"(d) class 1/2 samples with own-class proba below class mean and another class on top: {len(flagged)} "
          f"of {m.sum()}; permutation-pair members flagged: {[i for i in flagged if any(i in p for p in PAIRS)]}")
    res["d"] = {"flagged_1_2": flagged, "thresholds": thr.tolist()}

    stage2 = a_ok and (b_ok or c_ok)
    print(f"stage 1 decision (a and (b or c)): {stage2}")
    res["stage2"] = stage2

    # ---------- stage 2: 5-class, nested weight selection ----------
    if stage2:
        SO = np.hstack([S, O])
        out = {}
        for name, F in [("stats36", S), ("stats36+order30", SO)]:
            f_raw, f_w, ws, cms, cls = [], [], [], [], []
            for r in range(10):
                pred_raw = np.zeros(len(y), int); pred_w = np.zeros(len(y), int)
                for tr, va in StratifiedKFold(5, shuffle=True, random_state=r).split(F, y):
                    w = choose_weight(y[tr], oof_proba_inner(F[tr], y[tr], 100 + r))
                    P = model().fit(F[tr], y[tr]).predict_proba(F[va])
                    pred_raw[va] = P.argmax(1)
                    Q = P.copy(); Q[:, 1] *= w
                    pred_w[va] = Q.argmax(1)
                    ws.append(w)
                f_raw.append(macro_f1(y, pred_raw)); f_w.append(macro_f1(y, pred_w))
                cms.append(confusion_matrix(y, pred_w, labels=LABELS))
                cls.append(f1_score(y, pred_w, labels=LABELS, average=None, zero_division=0))
            out[name] = {"raw": f_raw, "nested_weight": f_w, "weights": ws,
                         "class_f1": np.mean(cls, 0).round(3).tolist(), "confusion_sum": np.sum(cms, 0).tolist()}
            print(f"[{name}] raw {np.mean(f_raw):.3f} +/- {np.std(f_raw):.3f}, nested weight {np.mean(f_w):.3f} "
                  f"+/- {np.std(f_w):.3f}, class F1 {out[name]['class_f1']}, weights chosen "
                  f"{dict(zip(*np.unique(ws, return_counts=True)))}")
        dd = np.array(out["stats36+order30"]["nested_weight"]) - np.array(out["stats36"]["nested_weight"])
        adopt = dd.mean() >= 0.02
        print(f"stage 2: nested-weight Macro-F1 diff {dd.mean():+.3f} (positive in {(dd > 0).sum()}/10) "
              f"-> adopt (>= +0.02) {adopt}")
        res["stage2_results"] = out
        res["stage2_adopt"] = bool(adopt)

    with open("exp_order_results.json", "w") as f:
        json.dump(res, f, indent=2, default=lambda v: v.item() if hasattr(v, "item") else str(v))


def oof_proba_inner(F, y, seed):
    return oof_proba(F, y, seed, 5)


if __name__ == "__main__":
    main()
