"""사후 실험: 파생변수 / 모델 계열(HGB vs RF vs GB) / RF+GB 앙상블 / 임계값 선택.

기존 train.py와 같은 분할(StratifiedKFold 5, random_state=42+r, 10회 반복)을 쓴다.
test는 쓰지 않는다.

사전 채택 규칙(결과 보기 전에 정함): 반복 단위 짝지은 차이의 평균 > 0 이고
10회 중 8회 이상 개선이면 "개선", 반대 방향 8회 이상이면 "악화", 그 외 "근거 부족".

지표
- f1_05: 각 반복의 OOF 확률 전체에 임계값 0.5를 적용한 양성 F1
- f1_nested: 바깥 fold마다 학습 부분 안에서 내부 3-fold OOF로 임계값을 골라
  바깥 검증 부분에 적용한 뒤, 반복 단위로 모아 계산한 양성 F1 (선택 절차까지 평가 바깥)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

from train import CAT_COL, DERIVED, RAW_NUM, TARGET, add_derived

HERE = Path(__file__).parent
OUT = HERE / "posthoc_model_family_reps.csv"
N_REPEATS, SEED0 = 10, 42
THRESHOLDS = np.arange(0.05, 0.96, 0.01)

df = add_derived(pd.read_csv(HERE / "train.csv"))
df["grade_code"] = df[CAT_COL].cat.codes
y = df[TARGET].values

RAW_CODE = RAW_NUM + ["grade_code"]
DER_CODE = RAW_CODE + DERIVED
DER_CAT = RAW_NUM + [CAT_COL] + DERIVED                      # HGB 네이티브 범주형
FINAL_HGB_COLS = [c for c in DER_CAT if c != "process_temp_c"]  # 기존 최종 제출 설정


def rf(s):
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=300, random_state=s, n_jobs=-1))


def gb(s):
    return make_pipeline(SimpleImputer(strategy="median"), GradientBoostingClassifier(random_state=s))


def hgb(**kw):
    return lambda s: HistGradientBoostingClassifier(random_state=s, categorical_features="from_dtype", **kw)


# name: (builder, columns, nested 임계값 계산 여부)
MODELS = {
    "rf_raw": (rf, RAW_CODE, False),
    "gb_raw": (gb, RAW_CODE, False),
    "hgb_raw": (hgb(), RAW_NUM + [CAT_COL], False),
    "rf_der": (rf, DER_CODE, True),
    "gb_der": (gb, DER_CODE, True),
    "hgb_der": (hgb(), DER_CAT, False),
    "hgb_final": (hgb(class_weight="balanced"), FINAL_HGB_COLS, True),
    "hgb_leaf8": (hgb(max_leaf_nodes=8), DER_CAT, False),
}
ENSEMBLE = ("rf_der", "gb_der")


def best_threshold(y_true, p):
    f1s = [f1_score(y_true, (p >= t).astype(int), zero_division=0) for t in THRESHOLDS]
    return THRESHOLDS[int(np.argmax(f1s))]


def inner_oof(build, X, y_tr, seed):
    p = np.zeros(len(y_tr))
    for tr, va in StratifiedKFold(3, shuffle=True, random_state=seed).split(X, y_tr):
        p[va] = build(seed).fit(X.iloc[tr], y_tr[tr]).predict_proba(X.iloc[va])[:, 1]
    return p


def run_repeat(r):
    seed = SEED0 + r
    names = list(MODELS) + ["ens_der"]
    oof = {n: np.zeros(len(y)) for n in names}
    nested_pred = {n: np.zeros(len(y), dtype=int) for n in ["rf_der", "gb_der", "hgb_final", "ens_der"]}
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(df, y):
        inner = {}
        for name, (build, cols, nested) in MODELS.items():
            X = df[cols]
            oof[name][va] = build(seed).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
            if nested:
                inner[name] = inner_oof(build, X.iloc[tr], y[tr], seed)
        oof["ens_der"][va] = (oof[ENSEMBLE[0]][va] + oof[ENSEMBLE[1]][va]) / 2
        inner["ens_der"] = (inner[ENSEMBLE[0]] + inner[ENSEMBLE[1]]) / 2
        for name in nested_pred:
            t = best_threshold(y[tr], inner[name])
            nested_pred[name][va] = (oof[name][va] >= t).astype(int)
    row = {"repeat": r}
    for name in names:
        row[f"{name}__f1_05"] = f1_score(y, (oof[name] >= 0.5).astype(int))
    for name, pred in nested_pred.items():
        row[f"{name}__f1_nested"] = f1_score(y, pred)
    return row


def main():
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
    for r in range(N_REPEATS):
        if len(done) and r in set(done["repeat"]):
            continue
        t0 = time.time()
        row = run_repeat(r)
        done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
        done.to_csv(OUT, index=False)
        print(f"repeat {r} done ({time.time() - t0:.0f}s)", flush=True)
    print(done.drop(columns="repeat").agg(["mean", "std"]).T.round(4).to_string())


if __name__ == "__main__":
    sys.exit(main())
