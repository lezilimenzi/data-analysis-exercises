"""submission_04: 파생변수 + RF 단독, 임계값 0.5, train 전체 재학습.

설정 근거: posthoc_model_family.py (rf_der). submission_03 채점 후 비교용으로 추가 채점 (test 기반 선택에 쓰지 않음).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from posthoc_model_family import DER_CODE, df, gb, rf, y
from train import CAT_COL, ID_COL, TARGET, add_derived

HERE = Path(__file__).parent
SEED = 42

test = add_derived(pd.read_csv(HERE / "test.csv"))
test["grade_code"] = test[CAT_COL].cat.codes

X, X_test = df[DER_CODE], test[DER_CODE]
proba = np.mean([b(SEED).fit(X, y).predict_proba(X_test)[:, 1] for b in (rf,)], axis=0)

sample = pd.read_csv(HERE / "sample_submission.csv")
pred = pd.Series((proba >= 0.5).astype(int), index=test[ID_COL])
sub = pd.DataFrame({ID_COL: sample[ID_COL], TARGET: pred.loc[sample[ID_COL]].values})

assert list(sub.columns) == [ID_COL, TARGET]
assert len(sub) == 2000 and sub[ID_COL].is_unique
assert set(sub[ID_COL]) == set(test[ID_COL])
assert set(sub[TARGET].unique()) <= {0, 1}
assert (test["grade_code"] >= 0).all()

sub.to_csv(HERE / "submission_04.csv", index=False)
print(f"saved submission_04.csv: 양성 {sub[TARGET].sum()}/{len(sub)}")
