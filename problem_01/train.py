import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_NUM = ["ambient_temp_c", "process_temp_c", "rotation_rpm", "torque_nm", "tool_usage_min"]
CAT_COL = "equipment_grade"
TARGET = "inspection_required"
ID_COL = "record_id"

DERIVED = ["power_w", "wear_x_torque", "temp_diff"]
MISSING_FLAGS = ["ambient_temp_c_missing", "torque_nm_missing", "tool_usage_min_missing"]

FEATURE_TIERS = ["raw", "raw_derived", "raw_derived_missing"]
FEATURE_SETS = {
    "raw": RAW_NUM + [CAT_COL],
    "raw_derived": RAW_NUM + [CAT_COL] + DERIVED,
    "raw_derived_missing": RAW_NUM + [CAT_COL] + DERIVED + MISSING_FLAGS,
}


def add_derived(df):
    df = df.copy()
    df["power_w"] = df["torque_nm"] * df["rotation_rpm"] * (2 * np.pi / 60)
    df["wear_x_torque"] = df["tool_usage_min"] * df["torque_nm"]
    df["temp_diff"] = df["process_temp_c"] - df["ambient_temp_c"]
    for col in ["ambient_temp_c", "torque_nm", "tool_usage_min"]:
        df[f"{col}_missing"] = df[col].isna().astype(int)
    df[CAT_COL] = df[CAT_COL].astype(pd.CategoricalDtype(categories=["A", "B", "C"]))
    return df


def make_hgb(random_state, class_weight=None, **hgb_kwargs):
    return HistGradientBoostingClassifier(random_state=random_state, class_weight=class_weight, **hgb_kwargs)


def make_logreg(numeric_cols, random_state):
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                           ("scale", StandardScaler())]), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), [CAT_COL]),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000, random_state=random_state))])


def build_model_fn(model_name, feature_cols, class_weight=None, **hgb_kwargs):
    numeric_cols = [c for c in feature_cols if c != CAT_COL]

    def _build(random_state):
        if model_name == "logreg":
            return make_logreg(numeric_cols, random_state=random_state)
        if model_name == "hgb":
            return make_hgb(random_state=random_state, class_weight=class_weight, **hgb_kwargs)
        raise ValueError(model_name)

    return _build


class SeedEnsemble:
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)
        return self

    def predict_proba(self, X):
        probas = np.mean([m.predict_proba(X)[:, 1] for m in self.models], axis=0)
        return np.column_stack([1 - probas, probas])


def make_seed_ensemble_fn(base_build_fn, n_seeds=10):
    def _build(random_state):
        models = [base_build_fn(random_state=random_state * 1000 + i) for i in range(n_seeds)]
        return SeedEnsemble(models)

    return _build


def repeated_oof_eval(build_fn, X, y, n_splits=5, n_repeats=10, seed0=42):
    results = []
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed0 + r)
        oof_proba = np.zeros(len(y))
        for tr_idx, va_idx in skf.split(X, y):
            model = build_fn(random_state=seed0 + r)
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            oof_proba[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        thresholds = np.arange(0.05, 0.96, 0.01)
        f1s = [f1_score(y, (oof_proba >= t).astype(int), zero_division=0) for t in thresholds]
        best_i = int(np.argmax(f1s))
        results.append((thresholds[best_i], f1s[best_i]))
    return results


def summarize(results):
    thresholds = np.array([t for t, _ in results])
    f1s = np.array([f for _, f in results])
    return {"mean_f1": f1s.mean(), "std_f1": f1s.std(), "mean_threshold": thresholds.mean()}


def choose_for_model(model_name, stats_by_config):
    current_fs = FEATURE_TIERS[0]
    current = stats_by_config[(current_fs, model_name)]
    for fs in FEATURE_TIERS[1:]:
        candidate = stats_by_config[(fs, model_name)]
        if candidate["mean_f1"] - current["mean_f1"] > current["std_f1"]:
            current_fs, current = fs, candidate
    return current_fs, current


def main():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample_submission = pd.read_csv("sample_submission.csv")

    train = add_derived(train)
    test = add_derived(test)
    y = train[TARGET]

    configs = [(fs, m) for fs in FEATURE_TIERS for m in ["logreg", "hgb"]]
    stats_by_config = {}

    print("config comparison (5-fold x 10 repeats, OOF-best threshold per repeat)")
    print(f"{'feature_set':<22}{'model':<10}{'mean_f1':>10}{'std_f1':>10}{'mean_thr':>10}{'sec':>8}")
    for fs, model_name in configs:
        feature_cols = FEATURE_SETS[fs]
        X = train[feature_cols]
        build_fn = build_model_fn(model_name, feature_cols)
        t0 = time.time()
        results = repeated_oof_eval(build_fn, X, y)
        stats = summarize(results)
        stats_by_config[(fs, model_name)] = stats
        elapsed = time.time() - t0
        print(f"{fs:<22}{model_name:<10}{stats['mean_f1']:>10.4f}{stats['std_f1']:>10.4f}"
              f"{stats['mean_threshold']:>10.3f}{elapsed:>8.1f}")

    candidates = {m: choose_for_model(m, stats_by_config) for m in ["logreg", "hgb"]}
    print("\nper-model chosen feature set (simplicity-preferred within 1 std):")
    for model_name, (fs, stats) in candidates.items():
        print(f"  {model_name}: {fs} (mean_f1={stats['mean_f1']:.4f})")

    winner_model = max(candidates, key=lambda m: candidates[m][1]["mean_f1"])
    winner_fs, winner_stats = candidates[winner_model]
    print(f"\noverall winner: feature_set={winner_fs}, model={winner_model}, "
          f"mean_f1={winner_stats['mean_f1']:.4f}, threshold={winner_stats['mean_threshold']:.3f}")

    feature_cols = FEATURE_SETS[winner_fs]

    # follow-up simplification check (not part of the main grid): permutation
    # importance on the winning config showed process_temp_c contributes ~0
    # (redundant with ambient_temp_c + temp_diff, corr 0.88). Test dropping it.
    if "process_temp_c" in feature_cols:
        reduced_cols = [c for c in feature_cols if c != "process_temp_c"]
        reduced_build_fn = build_model_fn(winner_model, reduced_cols)
        reduced_results = repeated_oof_eval(reduced_build_fn, train[reduced_cols], y)
        reduced_stats = summarize(reduced_results)
        print(f"\nfollow-up: {winner_fs} minus process_temp_c -> "
              f"mean_f1={reduced_stats['mean_f1']:.4f}, std_f1={reduced_stats['std_f1']:.4f}")
        if reduced_stats["mean_f1"] - winner_stats["mean_f1"] > -winner_stats["std_f1"]:
            print("  not meaningfully worse -> dropping process_temp_c (simpler)")
            feature_cols = reduced_cols
            winner_stats = reduced_stats
        else:
            print("  meaningfully worse -> keeping process_temp_c")

    # class_weight='balanced': CV showed identical mean_f1 to class_weight=None
    # on this feature set (tied), but graded test F1 favored balanced
    # (submission_02 0.8372 vs submission_01 0.8226). Adopted per user
    # decision despite the CV tie (see EXPERIMENTS.md Run 3 / submission log).
    final_class_weight = "balanced" if winner_model == "hgb" else None
    if final_class_weight:
        cw_build_fn = build_model_fn(winner_model, feature_cols, class_weight=final_class_weight)
        cw_results = repeated_oof_eval(cw_build_fn, train[feature_cols], y)
        winner_stats = summarize(cw_results)
        print(f"\nclass_weight='balanced' (adopted per user decision) -> "
              f"mean_f1={winner_stats['mean_f1']:.4f}, threshold={winner_stats['mean_threshold']:.3f}")

    # hyperparameter follow-up (not a full grid): a few curated HGB variants,
    # compared against the current default via the same repeated CV, on top
    # of the adopted feature set + class_weight.
    if winner_model == "hgb":
        hp_candidates = {
            "default": {},
            "regularized": {"min_samples_leaf": 40, "max_leaf_nodes": 15},
            "slow_learn": {"learning_rate": 0.05, "max_iter": 300},
            "l2": {"l2_regularization": 1.0},
        }
        print("\nhyperparameter comparison (class_weight='balanced' fixed):")
        hp_stats = {}
        for name, kwargs in hp_candidates.items():
            hp_build_fn = build_model_fn(winner_model, feature_cols, class_weight=final_class_weight, **kwargs)
            hp_results = repeated_oof_eval(hp_build_fn, train[feature_cols], y)
            hp_stats[name] = summarize(hp_results)
            print(f"  {name:<12} mean_f1={hp_stats[name]['mean_f1']:.4f} "
                  f"std_f1={hp_stats[name]['std_f1']:.4f} threshold={hp_stats[name]['mean_threshold']:.3f}")

        best_hp = max(hp_stats, key=lambda k: hp_stats[k]["mean_f1"])
        default_stats = hp_stats["default"]
        if hp_stats[best_hp]["mean_f1"] - default_stats["mean_f1"] > default_stats["std_f1"]:
            chosen_hp = best_hp
        else:
            chosen_hp = "default"
        print(f"  chosen: {chosen_hp} (simplicity-preferred within 1 std of default)")
        final_hp_kwargs = hp_candidates[chosen_hp]
        winner_stats = hp_stats[chosen_hp]
    else:
        final_hp_kwargs = {}

    base_build_fn = build_model_fn(winner_model, feature_cols, class_weight=final_class_weight, **final_hp_kwargs)

    # seed ensemble follow-up: average predict_proba across multiple random
    # seeds to reduce prediction variance (no new information, pure variance
    # reduction). Verified via the same repeated CV before adopting.
    use_seed_ensemble = False
    if winner_model == "hgb":
        ensemble_build_fn = make_seed_ensemble_fn(base_build_fn, n_seeds=10)
        ensemble_results = repeated_oof_eval(ensemble_build_fn, train[feature_cols], y)
        ensemble_stats = summarize(ensemble_results)
        print(f"\nseed ensemble (n=10) -> mean_f1={ensemble_stats['mean_f1']:.4f} "
              f"std_f1={ensemble_stats['std_f1']:.4f} threshold={ensemble_stats['mean_threshold']:.3f}")
        if ensemble_stats["mean_f1"] - winner_stats["mean_f1"] > winner_stats["std_f1"]:
            print("  meaningfully better -> adopting seed ensemble for final model")
            use_seed_ensemble = True
            winner_stats = ensemble_stats
        else:
            print("  not meaningfully better -> keeping single-seed model (simpler)")

    if use_seed_ensemble:
        final_model = make_seed_ensemble_fn(base_build_fn, n_seeds=10)(random_state=42)
    else:
        final_model = base_build_fn(random_state=42)
    final_model.fit(train[feature_cols], y)

    proba_test = final_model.predict_proba(test[feature_cols])[:, 1]
    pred_test = (proba_test >= winner_stats["mean_threshold"]).astype(int)

    submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET: pred_test})
    assert submission[ID_COL].equals(sample_submission[ID_COL]), "ID order mismatch with sample_submission"
    assert set(submission[TARGET].unique()) <= {0, 1}
    assert len(submission) == 2000 and submission[ID_COL].is_unique

    submission.to_csv("submission.csv", index=False)
    print(f"\nsaved submission.csv, predicted positive rate = {pred_test.mean():.4f}")


if __name__ == "__main__":
    main()
