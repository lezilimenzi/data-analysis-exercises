# PLAN.md — 야전 정비지원 장비 점검 분류 (구현 계획)

> 상태: **계획 초안. 구현 코드 작성 전.**
> 선행 문서: `RESEARCH.md` (EDA 결과). 이 문서는 RESEARCH.md 7절 "모델링 시사점"과 대화에서 합의한 전략을 구체적인 구현 설계로 옮긴 것이다.
> **승인 전까지 구현하지 않는다.**

## 0. 전략 요약 (합의된 내용)

| 항목 | 결정 |
|---|---|
| 검증 | Stratified K-fold 반복 CV, OOF 확률로 평가 |
| 전처리 | equipment_grade 인코딩, 결측치는 트리 모델 NaN 네이티브 처리 (대체 불필요) |
| 모델 | 로지스틱 회귀(베이스라인) vs HistGradientBoostingClassifier(메인) |
| 피처 | raw 6개 → raw+파생(power_w, wear_x_torque, temp_diff) → raw+파생+결측 지시변수, **3단계로 비교** (파생변수와 결측지시변수는 근거 수준이 달라 분리, 완전 조합 탐색은 안 함) |
| 임계값 | OOF 확률에서 F1 최대화 지점 탐색 (0.5 고정 아님) |
| 불균형 대응 | 임계값 조정 우선, 부족 시 `class_weight='balanced'` 2차 검토 |
| 최종 | 확정 설정으로 train 전체 재학습 → test 예측 → submission.csv |

**환경 확인 완료**: pandas 2.3.2, scikit-learn 1.7.2 설치됨. `HistGradientBoostingClassifier`는 `class_weight` 파라미터 지원 확인 (`inspect.signature`로 검증), `categorical_features='from_dtype'`가 기본값이라 pandas `category` dtype 컬럼을 자동으로 범주형 처리함 — 이 모델에는 원-핫 인코딩이 불필요.

## 1. 파일 구성

| 파일 | 역할 |
|---|---|
| `train.py` (신규) | 데이터 로드 → CV 비교 → 최종 학습 → `submission.csv` 저장까지 한 스크립트 |
| `submission.csv` (신규, 출력) | 제출 파일 |

단일 스크립트로 유지한다 (Simplicity First — 이 규모에서 여러 파일로 쪼갤 이유 없음). 함수 단위로 구분하되 CLI 옵션 등 불필요한 유연성은 추가하지 않는다.

## 2. 데이터 로드 & 컬럼 정의

```python
import pandas as pd

RAW_NUM = ["ambient_temp_c", "process_temp_c", "rotation_rpm", "torque_nm", "tool_usage_min"]
CAT_COL = "equipment_grade"
TARGET = "inspection_required"
ID_COL = "record_id"

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
```

- BOM/구분자는 `read_csv` 기본값으로 처리됨 (RESEARCH.md에서 이미 확인).
- `equipment_grade`는 `category` dtype으로 캐스팅 (HGB의 `from_dtype` 자동 인식용).

## 3. 파생 변수 & 결측 지시변수 정의

```python
def add_derived(df):
    df = df.copy()
    df["power_w"] = df["torque_nm"] * df["rotation_rpm"] * (2 * 3.141592653589793 / 60)
    df["wear_x_torque"] = df["tool_usage_min"] * df["torque_nm"]
    df["temp_diff"] = df["process_temp_c"] - df["ambient_temp_c"]
    for col in ["ambient_temp_c", "torque_nm", "tool_usage_min"]:
        df[f"{col}_missing"] = df[col].isna().astype(int)
    return df
```

- 결측이 있는 원본 열(ambient, torque, tool_usage)에서 계산되는 파생 변수(power_w, wear_x_torque, temp_diff)는 입력이 결측이면 자동으로 NaN이 된다 — 별도 처리 불필요(HGB가 NaN 처리), 로지스틱 회귀 경로는 4절의 Imputer가 이 파생 변수에도 동일하게 적용된다.

## 4. 피처셋 정의 (3단계로 비교, 완전 조합 탐색은 안 함)

```python
DERIVED = ["power_w", "wear_x_torque", "temp_diff"]
MISSING_FLAGS = ["ambient_temp_c_missing", "torque_nm_missing", "tool_usage_min_missing"]

FEATURE_SETS = {
    "raw": RAW_NUM + [CAT_COL],
    "raw_derived": RAW_NUM + [CAT_COL] + DERIVED,
    "raw_derived_missing": RAW_NUM + [CAT_COL] + DERIVED + MISSING_FLAGS,
}
```

파생 변수(power_w, wear_x_torque, temp_diff)와 결측 지시변수는 근거 수준이 다르다 — 파생변수는 물리적 근거(power = 토크×각속도)와 EDA 신호가 뚜렷한 반면, 결측 지시변수는 RESEARCH.md 3절에서 "결측 자체가 라벨과 관련 있다는 통계적 근거는 현재 없다"고 이미 밝힌 약한 가설이다. 그래서 하나로 묶지 않고 **raw → raw+파생 → raw+파생+결측지시변수** 순서로 단계적으로 추가해서, 두 그룹의 효과가 섞여 서로 가리지 않도록 한다. 다만 raw/파생/결측 3개를 넘어서는 완전 조합(예: raw+결측지시변수만, 파생 없이)까지는 탐색하지 않는다 — config 수를 3×2=6개로 제한하기 위함.

## 5. 전처리 & 모델 파이프라인

**HistGradientBoostingClassifier**: 전처리 없이 DataFrame을 그대로 입력 (category dtype 컬럼은 자동 인식, 수치형 NaN은 네이티브 처리).

```python
from sklearn.ensemble import HistGradientBoostingClassifier

def make_hgb(class_weight=None, random_state=42):
    return HistGradientBoostingClassifier(random_state=random_state, class_weight=class_weight)
```

**로지스틱 회귀 (베이스라인)**: OneHot + Impute(median) + Scale이 필요.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def make_logreg(numeric_cols, random_state=42):
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                           ("scale", StandardScaler())]), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), [CAT_COL]),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000, random_state=random_state))])
```

`numeric_cols`는 피처셋에 따라 RAW_NUM만 쓸지, 파생/지시변수까지 포함할지로 달라진다 (`FEATURE_SETS`에서 CAT_COL을 뺀 나머지).

## 6. 검증 절차 (Repeated Stratified K-fold + OOF 임계값 탐색)

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np

def repeated_oof_eval(build_model_fn, X, y, n_splits=5, n_repeats=10, seed0=42):
    results = []  # (repeat_idx, best_threshold, best_f1)
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed0 + r)
        oof_proba = np.zeros(len(y))
        for tr_idx, va_idx in skf.split(X, y):
            model = build_model_fn(random_state=seed0 + r)
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            oof_proba[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        thresholds = np.arange(0.05, 0.96, 0.01)
        f1s = [f1_score(y, (oof_proba >= t).astype(int), zero_division=0) for t in thresholds]
        best_i = int(np.argmax(f1s))
        results.append((thresholds[best_i], f1s[best_i]))
    return results
```

- fold 개수: 5, 반복: 10회 (총 50회 fold 학습/config당). 8,000행 규모에서 CPU로 충분히 빠르게 처리 가능.
- 각 반복마다 **그 반복 전용 OOF 확률**로 임계값을 탐색한다 (한 반복 안에서는 검증 데이터 정답을 학습에 쓰지 않음 — problem.md 4절 요건 충족).
- 반복 간 `best_f1`의 평균/표준편차로 "설정이 안정적으로 좋은지"를 판단한다 (RESEARCH.md 2절에서 지적한 단일 분리의 분산 문제 완화).

## 7. 비교 & 선택

```python
configs = [
    ("raw", "logreg"), ("raw", "hgb"),
    ("raw_derived", "logreg"), ("raw_derived", "hgb"),
    ("raw_derived_missing", "logreg"), ("raw_derived_missing", "hgb"),
]
# 각 config에 대해 repeated_oof_eval 실행 → mean_f1, std_f1, mean_threshold 출력/표로 정리
# 선택 기준: mean_f1이 가장 높은 config를 1차 후보로 하되,
#            더 단순한 feature set과의 mean_f1 차이가 표준편차 이내로 작으면 더 단순한 쪽을 선택
#            (raw_derived_missing이 raw_derived보다 유의하게 낫지 않으면 raw_derived 채택, 이런 식)
```

- 6개 config를 모두 출력하는 비교표를 만든다 (config, mean_f1, std_f1, mean_threshold).
- **불균형 2차 대응**: 선택된 config의 mean_f1이 낮다고 판단되면(임의 고정 기준 없이, 비교표를 보고 판단), 같은 feature set에 `class_weight='balanced'`를 추가한 hgb 변형 1개만 추가로 같은 절차로 평가한다. 이는 그리드에 넣지 않고 필요할 때만 순차적으로 시도한다 (전략 5번, 다중비교 억제 원칙 유지).
- **후속 단순화 검증 (Run 2에서 추가, EXPERIMENTS.md 참조)**: 메인 그리드에서 뽑힌 승자 config에 대해, permutation importance 등 진단으로 기여가 거의 없어 보이는 변수(예: process_temp_c, importance_mean이 importance_std보다 작음)를 하나 제거한 변형을 동일한 반복 CV로 재검증한다. mean_f1 차이가 표준편차 이내면(유의미하게 나쁘지 않으면) 더 단순한 쪽(제거)을 최종 채택한다. 이것도 그리드가 아니라 승자에 대한 조건부 순차 검증이다.
- **예외: class_weight='balanced' 채택 경로 (Run 3, submission_01/02 채점 이후)**: CV에서는 class_weight=None과 'balanced'의 mean_f1이 완전히 동일(0.8311)해서 CV 기준으로는 구분할 근거가 없었다. 그러나 실제 공식 채점에서 submission_02(balanced)가 submission_01(None)보다 F1 0.0146 높게 나왔고, 사용자가 이 결과를 근거로 balanced를 최종 채택하기로 **명시적으로 결정**했다. 이는 이 계획의 CV 기반 선택 기준(표준편차 이내면 단순한 쪽 채택)을 사용자 판단으로 오버라이드한 예외이며, train.py에도 그 경위를 주석으로 남겨뒀다. test 표본이 작아(양성 68건, 두 제출 간 예측이 다른 행 7개) 이 차이가 노이즈일 가능성도 있다는 점은 EXPERIMENTS.md에 기록해둠.
- **하이퍼파라미터 후속 검증 (Run 4)**: class_weight='balanced' 채택 이후, HGB 하이퍼파라미터 4개 후보(default/regularized/slow_learn/l2)를 동일 CV로 비교했으나 default 대비 유의미한 개선 없음 → default 유지.

## 8. 최종 학습 & 예측 & 제출

```python
# 1. 선택된 (feature_set, model) 로 train 전체(8,000행) 재학습
# 2. 최종 임계값 = 선택된 config의 반복별 best_threshold 평균
# 3. test 예측 확률 → 임계값 적용 → 0/1 정수 변환
# 4. submission.csv 저장

X_full = add_derived(train)[chosen_feature_cols]
y_full = train[TARGET]
final_model = build_model_fn(random_state=42)
final_model.fit(X_full, y_full)

X_test = add_derived(test)[chosen_feature_cols]
proba_test = final_model.predict_proba(X_test)[:, 1]
pred_test = (proba_test >= final_threshold).astype(int)

submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET: pred_test})
submission.to_csv("submission.csv", index=False)
```

- test.csv 행 순서가 sample_submission과 동일함을 RESEARCH.md 1절에서 이미 확인했으므로 별도 정렬 불필요. 다만 저장 직전에 `submission[ID_COL].equals(sample_submission[ID_COL])`로 한 번 더 검증한다.
- `inspection_required`가 정수 0/1인지, 행 수 2,000, 중복/누락 없는지 확인한다 (problem.md 6절 요건).

## 9. 트레이드오프 / 가정

- **피처셋은 raw → raw_derived → raw_derived_missing 3단계로만 비교하고, 완전 조합 탐색은 하지 않는다** — 파생변수와 결측지시변수는 근거 수준이 달라 분리했지만, 그 이상(예: 결측지시변수만 단독)까지는 다중비교 억제를 위해 탐색하지 않는다. 개별 변수 효과가 궁금하면 최종 선택된 모델의 feature importance로 사후 확인한다(선택 사항, 그리드에는 포함 안 함).
- **Nested CV는 쓰지 않는다** — config가 6~7개로 적어서(다중비교 위험이 상대적으로 작음) 복잡도 대비 이득이 작다고 판단. 대신 반복 CV(10회)로 각 config의 분산을 직접 보여줘서 사람이 판단할 근거를 제공한다.
- **최종 임계값은 선택된 config의 반복별 최적 임계값 평균**을 쓴다 — 반복마다 최적 임계값이 다르면(예: 0.3~0.5 사이 요동) 평균이 개별 최댓값보다 더 안정적인 선택일 가능성이 높다는 가정. 반복 간 임계값 변동이 크면(표준편차가 크면) 비교표에서 드러나므로 결과를 보고 재검토한다.
- **equipment_grade 인코딩**: HGB는 category dtype 그대로, 로지스틱 회귀는 OneHot. 두 모델이 이 변수의 정보를 다르게 쓰는 셈이지만, 각 모델에 맞는 표준적인 방식이라 판단.
- **class_weight 변형은 그리드가 아니라 조건부 2차 시도** — 이유는 위 6절/7절에서 설명.

## 10. 검증 전략 (완료 기준)

1. `train.py` 실행 시 에러 없이 끝까지 실행됨
2. 비교표(4개 config의 mean_f1, std_f1, mean_threshold)가 출력됨 → 사람이 확인 가능
3. `submission.csv` 생성 확인:
   - 행 수 2,000, 컬럼 `record_id,inspection_required`
   - test.csv의 ID를 중복/누락 없이 포함
   - `inspection_required` 값이 {0, 1}만 존재 (정수형)
   - 행 순서가 sample_submission과 동일
4. 예측 양성 비율이 0%나 100%처럼 비상식적이지 않은지 확인 (train 양성률 3.39% 근방인지 참고, 단 test가 반드시 같을 필요는 없음)
5. (선택) 검증 데이터 정답이 전처리 대체값 계산에 쓰이지 않았는지 코드 리뷰로 재확인 (problem.md 4절 요건)

## 11. 미해결 / 확인 필요 사항

- `n_repeats=10`이 시간상 적절한지는 실행해보고 조정 가능 (너무 오래 걸리면 5로 축소, config가 6개로 늘어 총 fold 학습 횟수도 증가했음: 6 config × 10 repeat × 5 fold = 300회).
- `raw_derived`/`raw_derived_missing` 채택 여부, `class_weight` 필요 여부는 실행 결과(비교표)를 보고 결정 — 지금은 확정하지 않는다.
- 이 계획에 대한 수정 사항이 있으면 이 파일에 반영한 뒤 구현을 시작한다.
