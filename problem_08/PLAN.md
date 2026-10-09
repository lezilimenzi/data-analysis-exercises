# PLAN.md — 8번 구현 계획 (v1)

근거: RESEARCH.md. **승인 전에는 구현하지 않습니다.**

## 변경 이력

v1 검토 결과 (2026-10-05): 아래 결정이 모두 v1 내용과 같아 본문은 바꾸지 않았습니다.

| 검토 항목 | 결정 |
|---|---|
| #1 | B 포함 |
| #2 | test에서 A·B가 달라도 A 유지 |
| #3 | 정수 샘플 번호 사용 |
| #4 | 전체 train으로 경계 산출 |
| #5 | submission.csv를 utf-8-sig로 저장 |

## 1. 전략 요약

| 후보 | 내용 | 역할 |
|---|---|---|
| A | 9.0초 이후 압력이 처음 20 bar를 넘는 정수 샘플 번호 → train 클래스 범위의 중간점 경계 3개로 구간을 나눔 | 주 모델이자 안전본 |
| B | 9.0~10.0초 원시 압력 100개 값 → RandomForest(300, random_state=0) | 독립 교차 확인. 경계 하나가 아니라 파형 전체 모양으로 판단 |

- **약점 보완** (RESEARCH §4):
  - 0↔1 경계가 좁은 문제 → 100Hz 해상도를 그대로 씁니다. 다운샘플링이나 전역 통계는 쓰지 않습니다.
  - test 분포 이동 문제 → 무작위 fit/valid 외에 이동을 흉내 낸 분할 2종(온도 분할, 운전조건 군집 LOGO)으로 검증합니다.
  - 저압 이상 사이클 문제 → 낮은 고정 임계값(20 bar)을 쓰고, 여러 임계값 투표로 일관성을 점검합니다.
- **과적합 방지**:
  - 후보는 2개로 제한하고 결과를 보기 전에 정해 둡니다.
  - A의 경계는 최적화하지 않고 클래스 범위의 중간점으로 정합니다.
  - B는 기본 하이퍼파라미터를 쓰고 튜닝하지 않습니다.
  - test 결과를 보고 모델이나 경계를 바꾸지 않습니다.
- **채택하지 않는 것**: 온도 보정(효과 0.6샘플, 간격 4.8샘플), 임계값 최적화, 클래스 가중치, 의사 라벨링, MiniRocket·딥러닝·전체 신호 GBM, flow·온도·진동 특징 추가. 근거는 RESEARCH §2–§5입니다.

## 2. 채택 규칙 (CLAUDE.md 5조)

1. 세 가지 분할에서 A와 B를 짝지어 비교합니다: fit→valid, 온도 분할(양방향), 군집 LOGO.
2. 모두 동점(예상: 전부 1.0)이면 5조 (a)에 따라 **A를 채택**합니다.
3. B가 짝지은 비교에서 평균 양수이고 과반의 짝에서 앞서면 B를 채택합니다.
4. test 진단(아래 §4)은 정보 제공용입니다. 진단 결과로 제출 모델을 바꾸지 않습니다.

## 3. 시간 예산 (30분)

| 구간 | 누적 | 할 일 | 후퇴 규칙 |
|---|---|---|---|
| 0–3분 | 3 | 로드, `rise()` 함수 | — |
| 3–6분 | 6 | **A로 test 예측 → submission.csv 저장** (안전본) | 반드시 완료 |
| 6–11분 | 11 | 검증: fit→valid, 온도 분할, 군집 LOGO (A, B 짝지어) | 11분에 끝나지 않으면 A로 확정 |
| 11–14분 | 14 | test 진단: 임계값 투표 일치율, A·B 일치율, 경계 여유 분포 | 생략 가능. 제출에는 영향 없음 |
| 14–17분 | 17 | 채택 판정, 최종 submission.csv 형식 점검, 노트북 저장 | — |
| 17–30분 | — | 여유. 동점일 때 이른 제출이 유리하므로 일찍 제출 | — |

## 4. 구현 내용

### 파일

| 경로 | 작업 |
|---|---|
| `problem_08/problem.ipynb` | 새로 생성 (README가 이 파일명을 지정) |
| `problem_08/submission.csv` | 새로 생성. `cycle_id,label`, test.csv 순서, `utf-8-sig` 인코딩 (sample과 같게 BOM 포함) |

### 노트북 셀 구성 (의사코드)

```python
# 셀 1: 로드
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
d = np.load('data/sensor_data.npz'); ids = d['cycle_id']
tr = pd.read_csv('data/train.csv', encoding='utf-8-sig'); te = pd.read_csv('data/test.csv', encoding='utf-8-sig')
idx = {c: i for i, c in enumerate(ids)}
ti = tr.cycle_id.map(idx).values; si = te.cycle_id.map(idx).values; y = tr.label.values
P, T = d['pressure'], d['temperature']

# 셀 2: 후보 A
def rise(thr=20):                      # 9.0s(=900) 이후 처음 thr bar를 넘는 샘플 번호
    seg = P[:, 900:1200]
    return np.argmax(seg > thr, 1) + 900
def fit_bounds(r, mask):               # 클래스 범위 중간점
    rg = [(r[ti][mask & (y == k)].min(), r[ti][mask & (y == k)].max()) for k in range(4)]
    return [(rg[k][1] + rg[k+1][0]) / 2 for k in range(3)]
r = rise(20)
bounds = fit_bounds(r, np.ones(len(ti), bool))   # 전체 train → 예상 [940, 948.5, 959.5]
predA = np.digitize(r, bounds)

# 셀 3: 안전본 저장
pd.DataFrame({'cycle_id': te.cycle_id, 'label': predA[si]}).to_csv('submission.csv', index=False, encoding='utf-8-sig')

# 셀 4: 검증 분할 정의
fit = tr.split.values == 'fit'
tm = T.mean(1)
X_cond = 운전조건 6특징 (RESEARCH §3과 동일) → StandardScaler → KMeans(6, n_init=10, random_state=0) → g
splits = [('fit→valid', fit, ~fit),
          ('cool→hot', tm[ti] < 45, tm[ti] >= 50), ('hot→cool', tm[ti] >= 50, tm[ti] < 45)]
         + [(f'LOGO g{G}', g[ti] != G, g[ti] == G) for G in 군집 중 n>=20]

# 셀 5: A·B 짝지어 비교 → 표 (split, F1_A, F1_B, 최소여유_A)
W = P[:, 900:1000]
for name, trn, tst in splits:
    bA = fit_bounds(r, trn); fA = f1(y[tst], np.digitize(r[ti][tst], bA))
    B = RandomForestClassifier(300, n_jobs=-1, random_state=0).fit(W[ti][trn], y[trn]); fB = f1(y[tst], B.predict(W[ti][tst]))

# 셀 6: test 진단 (정보 제공용)
- 임계값 {5,10,20,40,80}별 예측이 A와 몇 % 일치하는지
- B(train 전체로 학습)와 A의 test 일치율, 불일치 cycle_id 목록
- test 경계 여유 분포, 여유 < 2샘플인 건수

# 셀 7: 채택 판정 (§2 규칙) → 최종 submission.csv 다시 저장 + 형식 점검
assert len(sub) == 441, (sub.cycle_id == te.cycle_id).all(), set(sub.label) ⊆ {0,1,2,3}
print(sub.label.value_counts())
```

## 5. 가정

- 20 bar 임계값은 EDA에서 정했습니다. 다만 5~120 bar 전 구간에서 예측이 같으므로 선택 편향은 사실상 없습니다.
- 전체 train으로 정한 경계와 fit만으로 정한 경계가 같습니다(RESEARCH §2). 최종 경계는 전체 train으로 정합니다.
- test에도 train과 같은 압력 재상승 패턴이 있습니다. 확인 결과 test 441개 모두 20 bar를 넘었습니다.

## 6. 검증 기준 (성공 조건)

| 단계 | 확인 |
|---|---|
| 안전본 | submission.csv 441행, ID 순서가 test.csv와 같음, label ∈ {0,1,2,3} |
| 검증 | 각 분할에서 A의 F1 표 출력. 기대값은 전부 1.0 |
| 진단 | 임계값 투표 일치율 100%, A·B 일치율 100%, 여유 < 2샘플 0건 (EDA 기준 기대값) |
| 마무리 | 노트북을 처음부터 끝까지 다시 실행해도 오류 없음, 저장 완료 |

## 7. 결론이 바뀔 조건

- 시험 환경에서 경계가 EDA와 다르게 나오면, 데이터가 바뀐 것이므로 진단 셀을 확인합니다.
- B가 이동 흉내 분할 중 하나에서 A보다 앞서면, §2의 3번 규칙에 따라 B를 검토합니다.
- 군집 LOGO에서 A의 F1이 1.0 미만이면, 그 분할의 오분류 사례를 확인하는 데 3분 이내로 씁니다.
