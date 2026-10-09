# EXPERIMENTS.md — 실험 기록

> 형식: 실행마다 결과를 append. PLAN.md를 수정한 이유는 여기 기록을 근거로 한다.

## Run 1 — PLAN.md 초안 그대로 (2026-09-21)

`train.py` 최초 실행. PLAN.md 7절 config 6개(3 feature tier × 2 model) 전부 비교, 계획대로 `class_weight` 변형은 시도하지 않음(필요 없어 보임).

### 결과 (5-fold × 10 repeat, OOF-best threshold)

| feature_set | model | mean_f1 | std_f1 | mean_threshold | 소요(초) |
|---|---|---:|---:|---:|---:|
| raw | logreg | 0.4699 | 0.0066 | 0.191 | 3.0 |
| raw | hgb | 0.7374 | 0.0101 | 0.315 | 9.2 |
| raw_derived | logreg | 0.4931 | 0.0052 | 0.235 | 3.7 |
| **raw_derived** | **hgb** | **0.8286** | **0.0053** | **0.347** | 10.4 |
| raw_derived_missing | logreg | 0.4926 | 0.0057 | 0.230 | 3.9 |
| raw_derived_missing | hgb | 0.8286 | 0.0053 | 0.347 | 10.9 |

**선택된 config**: `raw_derived` + `hgb` (mean_f1 0.8286, threshold 0.347)

전체 실행 시간 42초 (300+ fold 학습). 예상보다 빨라서 `n_repeats` 축소는 불필요.

### 관측

1. **HGB가 로지스틱 회귀를 압도** — raw 기준으로도 0.7374 vs 0.4699. RESEARCH.md 4절에서 예상한 대로(U자형/임계값형 패턴은 선형 모델이 못 잡음) 확인됨.
2. **파생 변수(power_w, wear_x_torque, temp_diff)가 뚜렷하게 도움됨** — hgb 기준 raw(0.7374) → raw_derived(0.8286)로 +0.09. logreg도 소폭 개선(0.4699→0.4931). 물리적 근거가 있는 파생변수 가설이 CV로 검증됨.
3. **결측 지시변수는 효과 없음** — raw_derived(0.8286)와 raw_derived_missing(0.8286)이 완전히 동일. RESEARCH.md 3절의 "결측-라벨 관계 근거 없음" 관측과 일치. 이번 3단계 분리 비교(파생 vs 결측을 나눈 것)가 실제로 의미 있었음 — 하나로 묶었으면 "raw_plus가 raw보다 낫다"까지만 알고, 그게 파생 덕분인지 결측지시변수 덕분인지 몰랐을 것.
4. **std_f1가 전반적으로 작음(0.005~0.01)** — 반복 CV 결과가 안정적. config 간 mean_f1 차이(raw_derived hgb 0.8286 vs raw hgb 0.7374)가 std보다 훨씬 커서, 우연이 아니라 실제 효과로 판단.
5. **class_weight 미시도 판단의 근거**: mean_f1 0.83 수준이고 threshold(0.347)가 이미 0.5보다 낮은 쪽으로 이동해 불균형에 대응하고 있어 보여서, 이번 라운드에서는 2차 시도를 생략함. 필요하면 다음 라운드에서 비교 가능.
6. **최종 제출**: raw_derived + hgb, threshold 0.347로 train 전체 재학습 → test 예측 → 양성 56/2000(2.8%) → `submission.csv` 저장 완료.

### PLAN.md 대비 변경 사항
없음 — 계획대로 실행, 계획대로 결과 해석. 이번 라운드에서는 PLAN.md 수정 불필요.

### 다음 라운드에서 시도해볼 것 (아직 실행 안 함)
- `class_weight='balanced'` on hgb/raw_derived — 지금 F1이 이미 높아서 우선순위는 낮음
- HGB 하이퍼파라미터(max_iter, max_leaf_nodes, learning_rate) 튜닝 — 시간이 남으면

## Run 2 — feature importance 진단 + process_temp_c 제거 검증 (2026-09-21)

### 진단: permutation importance (raw_derived + hgb, held-out 20%, n_repeats=20)

| feature | importance_mean | importance_std |
|---|---:|---:|
| rotation_rpm | 0.314 | 0.037 |
| temp_diff | 0.304 | 0.035 |
| power_w | 0.198 | 0.013 |
| wear_x_torque | 0.170 | 0.019 |
| equipment_grade | 0.079 | 0.021 |
| torque_nm | 0.063 | 0.010 |
| tool_usage_min | 0.033 | 0.021 |
| ambient_temp_c | 0.018 | 0.017 |
| process_temp_c | −0.002 | 0.014 |

**관측**:
- 파생변수 3개(temp_diff, power_w, wear_x_torque) 모두 held-out에서도 상위권 — Run 1의 CV 개선이 실제로 이 변수들에서 나옴을 재확인.
- `rotation_rpm`이 1위. RESEARCH.md 4절에서 단변량 AUC가 가장 낮았던(0.223) 변수인데, 다른 변수와 결합하면 가장 중요 — U자형 비단조 관계라 단변량 지표로는 신호가 안 보였을 뿐이라는 가설이 확인됨.
- `process_temp_c`는 importance_mean(−0.002)이 importance_std(0.014)보다 작아 **0과 통계적으로 구분 안 됨** — "성능을 깎는다"가 아니라 "고유 기여가 거의 없어 보인다"는 뜻. ambient_temp_c와 상관 0.88이고 temp_diff가 이미 그 차이를 뽑아내고 있어서 중복으로 추정.

### 검증: process_temp_c 제거 (raw_derived_no_process + hgb, 동일 CV 절차)

| config | mean_f1 | std_f1 | mean_threshold |
|---|---:|---:|---:|
| raw_derived (기존) | 0.8286 | 0.0053 | 0.347 |
| raw_derived − process_temp_c | 0.8311 | 0.0078 | 0.351 |

차이(+0.0025)가 표준편차 이내라 유의미한 차이 아님 → **Simplicity First 원칙에 따라 더 단순한 쪽(제거) 채택**. train.py에 "메인 그리드에서 뽑힌 승자에 대해 process_temp_c 제거를 조건부로 재검증"하는 후속 단순화 단계로 반영함 (class_weight와 같은 패턴 — 그리드에 넣지 않고 순차 검증).

### 최종 설정
`raw_derived − process_temp_c` + `hgb`, threshold 0.351, mean_f1(CV) 0.8311. train 전체 재학습 → test 예측 → 양성 56/2000(2.8%) → `submission.csv` 갱신. 결측치 있는 test 67행 모두 0/1 정상 예측 확인(NaN 없음).

### PLAN.md 대비 변경 사항
- PLAN.md 4절/8절에 "메인 그리드 승자에 대한 조건부 후속 단순화 검증(예: 낮은 permutation importance 변수 제거)" 단계를 추가함.

### 다음 라운드에서 시도해볼 것 (아직 실행 안 함)
- `class_weight='balanced'` on 최종 config — 우선순위 낮음
- HGB 하이퍼파라미터 튜닝 — 우선순위 낮음

### 제출 스냅샷: submission_01.csv (2026-09-21)

Run 2의 출력물(`raw_derived − process_temp_c` + `hgb`, threshold 0.351, CV mean_f1 0.8311)을 `submission_01.csv`로 이름 바꿔 채점용으로 보관.

### 공식 채점 결과 (2026-09-21, score.py 대조)

| 지표 | 값 |
|---|---:|
| F1 (공식) | 0.8226 |
| Precision | 0.9107 |
| Recall | 0.7500 |
| TP / FP / FN | 51 / 5 / 17 |

- CV 추정치(0.8311)와 실제 F1(0.8226) 차이는 0.0085로, CV std_f1(0.0078)과 비슷한 수준 → **CV가 실제 성능을 잘 예측함, 과적합 징후 없음.**
- Precision > Recall(0.91 vs 0.75) 관찰됨. 다만 이 수치 자체를 목표로 삼아 임계값/설정을 역산하면 test에 대한 사실상의 튜닝(problem.md 7절 금지 사항 취지 위반)이 되므로, 이 결과를 직접 최적화 대상으로 쓰지 않고 train 쪽 CV로만 다음 개선을 검증하기로 함.

## Run 3 — class_weight='balanced' 비교 (2026-09-21)

전략(RESEARCH.md 7절/PLAN.md 0절: "임계값 조정 우선, 부족 시 class_weight 2차 검토")에 따라, 하이퍼파라미터 튜닝보다 먼저 class_weight를 검증. 현재 승자 config(`raw_derived − process_temp_c` + `hgb`, class_weight=None)에 `class_weight='balanced'`만 추가한 변형 1개를 같은 CV(5-fold×10repeat)로 비교.

| config | mean_f1 | std_f1 | mean_threshold |
|---|---:|---:|---:|
| class_weight=None (기존) | 0.8311 | 0.0078 | 0.351 |
| class_weight='balanced' | 0.8311 | 0.0073 | 0.732 |

**관측**: mean_f1은 완전히 동일, mean_threshold만 0.351→0.732로 크게 이동. class_weight='balanced'가 학습 중 양성 샘플 손실 가중치를 높여 예측 확률 자체를 전반적으로 밀어올리지만, OOF 임계값을 매번 재탐색하기 때문에 최종 분류 결과(순위 기반 판정)는 사실상 동일해짐. 즉 **임계값 튜닝이 이미 불균형 대응 효과의 대부분을 흡수**하고 있었다는 RESEARCH.md 7절 가설이 CV로 확인됨.

**결론**: 개선 없음(차이 0) → Simplicity First에 따라 `class_weight=None`(기존, 파라미터 적음) 유지. train.py 변경 없음, 재제출 없음.

### PLAN.md 대비 변경 사항
없음 — class_weight는 계획대로 "부족 시 2차 검토"했고, 부족하지 않아 채택 안 함.

### 다음 후보
- HGB 하이퍼파라미터 튜닝(min_samples_leaf, max_leaf_nodes, learning_rate, l2_regularization) — class_weight로 효과가 없었으니 이제 시도해볼 차례. 우선순위는 있으나 mean_f1이 이미 0.83대로 높아 한계효용은 크지 않을 것으로 예상.

## 채점 결과: submission_02.csv (2026-09-21)

`class_weight='balanced'` 버전(threshold 0.732)을 `submission_02.csv`로 저장해 채점.

| 제출 | F1 (공식) |
|---|---:|
| submission_01 (class_weight=None) | 0.8226 |
| submission_02 (class_weight='balanced') | **0.8372** (+0.0146) |

**주의**: CV에서는 두 config의 mean_f1이 완전히 동일(0.8311)했다. 실제 차이는 예측이 다른 test 7행(2,000행 중)에서만 발생했고, test 양성이 68건뿐이라 이 정도 차이는 통계적 노이즈 범위 안에 있을 수 있다 — CV 증거만으로는 "balanced가 더 낫다"를 뒷받침하지 못한다. 그럼에도 사용자가 실제 채점 결과를 근거로 submission_02(class_weight='balanced')를 최종으로 채택하기로 결정함. 이 결정은 CV 규칙이 아니라 사용자의 명시적 판단에 따른 것으로 기록해둔다 (train.py에도 주석으로 남김).

## Run 4 — 하이퍼파라미터 비교 (2026-09-21)

채택된 config(`raw_derived − process_temp_c` + `hgb` + `class_weight='balanced'`)를 기준으로 HGB 하이퍼파라미터 4개 후보를 동일 CV(5-fold×10repeat)로 비교. 큰 그리드 대신 데이터 규모(8,000행, 양성 271건)에 영향이 클 만한 축(트리 복잡도, 학습률, L2) 하나씩만 기본값에서 바꾼 4개만 비교.

| config | 설정 | mean_f1 | std_f1 | threshold |
|---|---|---:|---:|---:|
| default | 전부 기본값 | 0.8311 | 0.0073 | 0.732 |
| regularized | min_samples_leaf=40, max_leaf_nodes=15 | 0.8161 | 0.0044 | 0.836 |
| slow_learn | learning_rate=0.05, max_iter=300 | 0.8346 | 0.0073 | 0.764 |
| l2 | l2_regularization=1.0 | 0.8288 | 0.0078 | 0.803 |

**관측**:
- `regularized`는 오히려 악화(트리를 너무 단순화하면 U자형/임계값형 패턴을 못 잡음 — RESEARCH.md 4절 관측과 일관)
- `slow_learn`이 가장 높지만(+0.0035) default의 std(0.0073)보다 작은 차이 → 선택 기준(표준편차 이내면 단순한 쪽)에 따라 **default 유지**
- 하이퍼파라미터 튜닝의 한계효용이 실제로 작다는 사전 예상이 CV로 확인됨

**결론**: 기본 하이퍼파라미터 유지. train.py를 전체 파이프라인(그리드 비교 → process_temp_c 제거 → class_weight 채택 → 하이퍼파라미터 비교 → 최종 학습/저장)으로 갱신, 실행 결과 `submission.csv`가 `submission_02.csv`와 완전히 동일함을 확인(diff 없음).

### PLAN.md 대비 변경 사항
- PLAN.md에 "class_weight는 CV 동률이었지만 실제 채점 결과 기준으로 사용자가 채택"이라는 예외적 의사결정 경로를 기록할 필요 있음 (다음 PLAN.md 갱신 시 반영).

### 다음 후보 (우선순위 낮음)
- 남은 개선 여지는 제한적으로 보임(하이퍼파라미터, class_weight 모두 CV 기준 한계효용 낮음). 추가로는 앙상블(logreg+hgb 확률 평균) 정도가 남은 후보.

## Run 5 — 시드 앙상블 검증 (2026-09-21)

채택된 config(`raw_derived − process_temp_c` + `hgb` + `class_weight='balanced'` + 기본 하이퍼파라미터)에 대해, seed 10개로 각각 학습한 뒤 확률을 평균하는 방식(`SeedEnsemble`)을 같은 CV(5-fold×10repeat, fold당 10개 시드 = 총 500회 학습)로 검증.

| config | mean_f1 | std_f1 | threshold |
|---|---:|---:|---:|
| 단일 시드 (기존) | 0.8311 | 0.0073 | 0.732 |
| 시드 앙상블 (n=10) | 0.8311 | 0.0073 | 0.732 |

**관측**: 소수점까지 완전히 동일. HGB의 `random_state`가 이 데이터/설정에서는 예측 결과에 사실상 영향을 주지 않는 것으로 보임 (early_stopping 검증 분할 등에서 오는 랜덤성이 미미하거나, 트리 구조가 시드와 무관하게 거의 동일하게 수렴하는 것으로 추정).

**결론**: 개선 없음 → 단일 시드 모델 유지(더 단순하고 5배 빠름). train.py, submission.csv 변경 없음(diff 확인).

### 지금까지 시도한 개선 경로 요약
- 파생변수 추가: 효과 큼 (+0.09, Run 1)
- 결측지시변수: 효과 없음 (Run 1)
- process_temp_c 제거: 효과 없음(더 단순해져서 채택, Run 2)
- class_weight='balanced': CV상 효과 없음, 실제 채점에서는 개선(+0.0146, 사용자 판단으로 채택)
- 하이퍼파라미터 튜닝: 효과 없음 (Run 4)
- 시드 앙상블: 효과 없음 (Run 5)

CV 기준으로 Run 1(파생변수) 이후로는 유의미한 개선이 없었음 — 현재 설정이 이 피처셋/모델 조합의 사실상 성능 상한으로 판단됨.

## Run 6 — 사후 실험: 모델 계열(HGB vs RF vs GB) · RF+GB 앙상블 · 임계값 선택 (2026-10-05)

**계기**: 다른 실험 기록에서 "파생변수 + RF·GB 앙상블" test F1 0.920635, "RandomForest" test F1 0.8976이 보고됨 (같은 test 2,000행·같은 채점기). 우리 최종(HGB) 0.8372보다 크게 높아 원인을 CV로 확인.
역산(양성 68건): 0.920635 = 58/63 → TP 58, FN 10, FP 0 / 0.8976 → TP 57, FN 11, FP 2.

**스크립트**: `posthoc_model_family.py` (반복별 점수 `posthoc_model_family_reps.csv`, 로그 `posthoc_model_family.log`)
- 분할: train.py와 동일 (StratifiedKFold 5, random_state=42+r, 10 repeat). test 미사용.
- 지표: `f1_05` = 반복별 OOF 전체에 임계값 0.5 / `f1_nested` = 바깥 fold의 학습 부분 안에서 내부 3-fold OOF로 임계값을 골라 바깥 검증 부분에 적용 (선택 절차까지 평가 바깥).
- 모델: RF(n_estimators=300, 중앙값 대체), GB(기본값, 중앙값 대체), HGB(네이티브 범주형/NaN). RF·GB는 장비 등급을 정수 코드로 입력.
- **사전 채택 규칙**(실행 전 확정): 짝지은 차이 평균 > 0 이고 10회 중 8회 이상 개선 → 개선 / 반대 8회 이상 → 악화 / 그 외 근거 부족.

### 결과 (평균 F1, 10 repeat)

| config | f1_05 | f1_nested |
|---|---:|---:|
| rf_raw | 0.7021 | |
| gb_raw | 0.7042 | |
| hgb_raw | 0.7115 | |
| rf_der | 0.8704 | 0.8680 |
| gb_der | 0.8692 | 0.8662 |
| hgb_der (기본) | 0.8188 | |
| hgb_leaf8 (max_leaf_nodes=8) | 0.8252 | |
| hgb_final (기존 최종: −process_temp_c, balanced) | 0.8107 | 0.8169 |
| **ens_der (RF·GB 확률 평균)** | **0.8723** | **0.8686** |

### 짝지은 비교

| 비교 | 평균 차이 | 개선/악화(/동점) | 판정 |
|---|---:|---|---|
| 원본 → +파생 (RF / GB / HGB) | +0.168 / +0.165 / +0.107 | 10/0 (세 모델 모두) | 개선 |
| hgb_final → rf_der (nested) | +0.051 | 10/0 | 개선 |
| hgb_final → rf_der / gb_der (0.5) | +0.060 / +0.059 | 10/0 | 개선 |
| hgb_der → hgb_leaf8 | +0.007 | 7/3 | 근거 부족 |
| hgb_leaf8 → gb_der | +0.044 | 10/0 | 개선 |
| gb_der → rf_der | +0.001 | 6/4 | 근거 부족 |
| rf_der → ens_der (0.5) | +0.0019 | 6/2/2 | 근거 부족 |
| rf_der → ens_der (nested) | +0.0006 | 5/5 | 근거 부족 |
| rf_der: 0.5 → nested 임계값 | −0.0024 | 0/9/1 | 악화 (폭 작음) |
| gb_der, ens_der: 0.5 → nested | −0.003, −0.004 | 5/5 | 근거 부족 |
| hgb_final: 0.5 → nested | +0.006 | 8/2 | 개선 |

**관측**:
- 파생변수는 이미 우리 풀이에 있었음 — 다른 실험과의 차이 원인이 아님. 세 모델 모두에서 효과 재확인.
- **차이의 원인은 모델 계열(HGB → RF/GB)**. 임계값 선택을 양쪽 다 평가 바깥으로 뺀 공정 비교에서도 +0.051, 10/10.
- HGB 잎 수를 GB 수준(8)으로 줄여도 GB와 +0.044 차이가 남음 → HGB 열세의 주원인은 트리 크기가 아님. 구간화(binning)·반복 수/학습률 등은 미검증.
- RF·GB 앙상블은 RF 단독과 구분되지 않음.
- 임계값 튜닝 효과는 모델 의존적: 확률이 치우친 HGB(balanced)에서만 이득, RF는 0.5가 이미 적절해 선택 노이즈만큼 손해.
- 기존 기록의 HGB 0.8311은 OOF-best 임계값(선택용) 점수라 이번 표의 값보다 높게 나옴. 이번 hgb_der는 범주형 처리 등 설정 차이로 Run 1의 raw_derived와도 약간 다름.
- Run 5 끝의 "현재 설정이 사실상 성능 상한"이라는 판단은 **틀렸음** — 첫 후보군을 logreg vs HGB로만 잡아 RF/GB 계열을 비교하지 않았기 때문.

## 채점 결과: submission_03.csv, submission_04.csv (2026-10-05)

- `make_submission_03.py`: 파생변수(+process_temp_c 포함) + RF·GB 확률 평균, 임계값 0.5, train 전체 재학습, random_state 42. **채점 전에** 사용자가 최종 후보로 확정.
- `make_submission_04.py`: 같은 설정의 RF 단독. submission_03 채점 후 비교용으로 사용자가 추가 채점 요청.

| 제출 | F1 (공식) | TP | FN | FP | precision | recall |
|---|---:|---:|---:|---:|---:|---:|
| submission_01 (HGB) | 0.8226 | 51 | 17 | 5 | 0.911 | 0.750 |
| submission_02 (HGB, balanced) | 0.8372 | 54 | 14 | 7 | 0.885 | 0.794 |
| **submission_03 (RF·GB 앙상블)** | **0.9147** | 59 | 9 | 2 | 0.967 | 0.868 |
| submission_04 (RF 단독) | 0.8976 | 57 | 11 | 2 | 0.966 | 0.838 |

**관측**:
- HGB(submission_02) → 앙상블: TP +5, FP −5. CV로 미리 확인한 방향(+0.05~0.06)과 일치 — test 기반 선택이 아님.
- CV 기대치(약 0.872)보다 +0.043 높음. 2,000행 단일 평가 SD(약 0.023, 25% holdout 20회 시뮬레이션, `posthoc_rf_gb_initial.py`) 기준 운이 좋은 쪽이지만 정상 변동 범위. 기대 성능은 0.87대로 본다.
- submission_03 vs 04: 예측이 다른 행은 **2개**뿐, 둘 다 실제 양성을 앙상블만 맞힘. 엇갈린 행 2개가 모두 한쪽에 유리할 확률은 우연만으로 50%(정확 이항, 양측 p=0.5) → 앙상블 우위의 근거로 쓰지 않음 (CV 결론 "근거 부족"과 일관).
- submission_04는 다른 실험의 RF 결과(0.8976, TP 57 / FN 11 / FP 2)와 혼동행렬까지 동일.

**최종 채택: submission_03 (파생변수 + RF·GB 앙상블)** — 사용자 결정 (2026-10-05). 채점 전에 확정한 후보를 그대로 유지한 것이며, RF 단독과의 test 차이(2행)는 채택 근거로 쓰지 않음.

### 지금까지 시도한 개선 경로 요약 (갱신)
- 파생변수 추가: 효과 큼 (Run 1, Run 6에서 RF·GB·HGB 모두 재확인)
- **모델 계열 HGB → RF/GB: 효과 큼 (+0.05~0.06, Run 6)**
- RF·GB 앙상블 vs RF 단독: CV·test 모두 구분 안 됨 (Run 6)
- 임계값 튜닝: HGB에서만 소폭 이득, RF에서는 불필요 (Run 6)
- 결측지시변수, 하이퍼파라미터, 시드 앙상블: 효과 없음 (Run 1, 4, 5)
