"""Paired differences for STRATEGY.md grades (added 2026-10-02).
Reuses train.py unchanged. Unit of pairing = repetition (10 repeats of 5-fold; each repetition scores the pooled
OOF with its own best threshold, exactly as train.py does). Asserts that each config reproduces the recorded mean F1.
Output: paired_recheck.csv (per comparison) and paired_recheck_reps.csv (per-repetition F1 of each config).
"""
import numpy as np
import pandas as pd

from train import CAT_COL, FEATURE_SETS, TARGET, add_derived, build_model_fn, repeated_oof_eval

train = add_derived(pd.read_csv("train.csv"))
y = train[TARGET]
no_proc = [c for c in FEATURE_SETS["raw_derived"] if c != "process_temp_c"]

CONFIGS = {  # name: (model, feature cols, class_weight, hgb kwargs, recorded mean F1 from EXPERIMENTS.md)
    "logreg_raw": ("logreg", FEATURE_SETS["raw"], None, {}, 0.4699),
    "hgb_raw": ("hgb", FEATURE_SETS["raw"], None, {}, 0.7374),
    "hgb_derived": ("hgb", FEATURE_SETS["raw_derived"], None, {}, 0.8286),
    "hgb_derived_missing": ("hgb", FEATURE_SETS["raw_derived_missing"], None, {}, 0.8286),
    "hgb_no_process": ("hgb", no_proc, None, {}, 0.8311),
    "hgb_no_process_balanced": ("hgb", no_proc, "balanced", {}, 0.8311),
    "hp_regularized": ("hgb", no_proc, "balanced", {"min_samples_leaf": 40, "max_leaf_nodes": 15}, 0.8161),
    "hp_slow_learn": ("hgb", no_proc, "balanced", {"learning_rate": 0.05, "max_iter": 300}, 0.8346),
    "hp_l2": ("hgb", no_proc, "balanced", {"l2_regularization": 1.0}, 0.8288),
}
reps = {}
for name, (model, cols, cw, kw, recorded) in CONFIGS.items():
    f1 = np.array([f for _, f in repeated_oof_eval(build_model_fn(model, cols, class_weight=cw, **kw), train[cols], y)])
    assert abs(f1.mean() - recorded) < 0.0005, (name, f1.mean(), recorded)
    reps[name] = f1
    print(f"{name}: mean {f1.mean():.4f} (recorded {recorded})", flush=True)

PAIRS = [("logreg_raw", "hgb_raw"), ("hgb_raw", "hgb_derived"), ("hgb_derived", "hgb_derived_missing"),
         ("hgb_derived", "hgb_no_process"), ("hgb_no_process", "hgb_no_process_balanced"),
         ("hgb_no_process_balanced", "hp_regularized"), ("hgb_no_process_balanced", "hp_slow_learn"),
         ("hgb_no_process_balanced", "hp_l2")]
rows = []
for a, b in PAIRS:
    d = reps[b] - reps[a]  # positive = candidate b better
    sd = d.std(ddof=1)
    rows.append(dict(baseline=a, candidate=b, n=len(d), mean=d.mean(), sd=sd,
                     mean_over_sd=d.mean() / sd if sd > 0 else np.nan,
                     better=int((d > 1e-12).sum()), worse=int((d < -1e-12).sum())))
out = pd.DataFrame(rows)
out.to_csv("paired_recheck.csv", index=False)
pd.DataFrame(reps).to_csv("paired_recheck_reps.csv", index_label="repeat")
print(out.round(4).to_string(index=False))
