# PLAN.md — 문제 6: SMS 스팸 이진 분류

> 작성일: 2026-10-01 · 상태: **검토 대기. Do not implement yet.**
> 근거: `RESEARCH.md`. 이전 문제의 설정값은 가져오지 않고 접근만 가져온다(`../STRATEGY.md`).
> 🔶 표시는 사용자 결정이 필요한 항목이다.

## 1. 목표와 성공 기준
- train 4000건으로 학습해 test 1000건의 label(0/1)을 예측한다.
- 지표: `f1_score(y, p, labels=[0,1], average="macro", zero_division=0)`
- 산출물
  - `problem.ipynb`: 데이터 로드부터 제출 파일 생성까지 한 번에 실행되고, 출력이 남은 상태로 저장
  - `submission.csv`: `message_id,label`, 1000행, test 순서, `utf-8-sig`(sample과 같이 BOM 포함)
- 성공 기준
  1. 노트북 전체 실행 시간 3분 이내.
  2. 형식 검사 통과(§6 셀 7).
  3. 최종 모델의 CV macro F1이 word NB 기준선(약 0.956)보다 확실히 높다.

## 2. 전략 요약
| 단계 | 내용 | 이유 |
|---|---|---|
| 기본 모델 | **char_wb(2-5) TF-IDF + LinearSVC(class_weight="balanced")** | 탐색한 계열 중 최고. 빠르고 규제가 단순하다 |
| 약점 보완 | 놓친 스팸(FN)이 오류의 대부분 → **결정 임계값을 내부 CV로 조정**. 후보로 word n-gram 추가(word+char) | FP가 거의 없어 경계를 스팸 쪽으로 약간 옮길 여유가 있다. 번호 없는 스팸의 어휘 신호를 word n-gram이 보탤 수 있다 |
| 과적합 방지 | ① 임계값은 바깥 fold 학습 부분 안에서만 선택(중첩) ② 후보 3개로 제한 ③ 연구에 쓰지 않은 새 분할 seed로 판정 ④ 사전 채택 규칙 미달이면 단순한 A 유지 | RESEARCH §7 결과는 이미 본 결과라 선택 편향이 있다 |
| 하지 않는 것 | 메타 특징·노이즈 필터링(RESEARCH에서 둘 다 악화), 하이퍼파라미터 격자 탐색, 트랜스포머·외부 데이터, 앙상블 | 시간 대비 이득 근거가 없거나 남은 오류(구독 콘텐츠·잘린 문장)를 텍스트로 맞출 수 없음 |

## 3. 후보 (고정, 탐색 없음)
```python
C = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)
W = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
A = make_pipeline(C, LinearSVC(C=1, class_weight="balanced"))                    # 안전 제출본
Cthr = TunedThresholdClassifierCV(A, scoring=macro_f1, cv=StratifiedKFold(3, shuffle=True, random_state=7), thresholds=41)
Ethr = TunedThresholdClassifierCV(make_pipeline(make_union(W, C), LinearSVC(C=1, class_weight="balanced")), 같은 설정)
```
- `LinearSVC`의 C=1, ngram 범위, `min_df=2`는 RESEARCH에서 쓴 고정값이다. 튜닝하지 않았으므로 최적값이라는 근거는 없다 [가정].

## 4. 검증 설계와 사전 채택 규칙
- `StratifiedKFold(5, shuffle=True, random_state=2026 + r)`, r = 0..2 → 15 fold. **RESEARCH에서 쓴 seed(42~44, 100~102)와 겹치지 않는다.**
- 세 후보를 같은 fold에서 학습·평가해 fold별 macro F1과 A 대비 짝지은 차이를 저장한다. pooled OOF 혼동 행렬도 출력한다.
- 템플릿 Group CV는 RESEARCH에서 층화 CV와 차이가 없었으므로 생략한다.

**채택 규칙** 🔶 (결과 보기 전에 고정)
| 조건 | 규칙 |
|---|---|
| 후보(Cthr 또는 Ethr) 채택 | A 대비 짝지은 차이 평균 > 0 **이고** 15 fold 중 개선 fold 수 > 악화 fold 수 + 3 |
| 둘 다 통과 | 짝지은 차이 평균이 큰 쪽. 차이가 0.001 이내면 더 단순한 Cthr |
| 둘 다 미달 | A 제출 |

- 개선 폭은 RESEARCH 기준으로도 fold 표준편차보다 작다. 이 규칙은 "방향이 일관된 작은 개선"만 받아들이는 운영 규칙이며, 유의성 판정이 아니다.

## 5. 시간 예산 (남은 약 19분 기준)
| 작업 | 예상 |
|---|---|
| 사용자 검토 | ~4분 |
| 노트북 작성 | ~4분 |
| 실행 (15 fold × 후보 3 + 최종 학습) | 약 1~2분 (RESEARCH 측정 기반 추정, 중첩 후보가 가장 무겁다) |
| 형식 점검·보고 | ~2분 |
| 여유 | ~5분 |

- **후퇴 순서**: 시간이 부족하면 셀 4(안전 제출본)만 실행한 결과를 제출한다. 그다음으로 Ethr를 빼고 A 대 Cthr만 비교한다.

## 6. 노트북 구성 (`problem.ipynb`)
1. import, `SEED`, 경로, `macro_f1` scorer
2. 데이터 로드(`utf-8-sig`), 행 수·결측·라벨 분포 출력 (핵심 EDA만)
3. 후보 정의 (§3)
4. **안전 제출본**: A를 전체 train에 학습 → `submission.csv` 먼저 저장
5. 15 fold 짝지은 비교 → 표(평균, std, A 대비 차이, 개선/악화 fold 수), pooled OOF 혼동 행렬, 놓친 스팸 예시 몇 개
6. §4 규칙으로 자동 선택 → 선택 모델을 전체 train에 재학습(임계값은 전체 train 내부 3-fold로 다시 선택) → `submission.csv` 덮어쓰기
7. 형식 검사: 행 수 1000, 컬럼 `["message_id","label"]`, ID 순서 = sample, label ∈ {0,1}, 결측 없음, 예측 스팸 비율 출력(train 12.5%와 크게 다르면 경고)

실행: `jupyter nbconvert --to notebook --execute --inplace problem.ipynb`

## 7. 가정과 위험
- test 라벨 분포가 train과 비슷하다고 가정한다. 다르면 임계값 조정의 효과가 달라질 수 있다.
- 남은 FN의 상당수는 텍스트로 구분할 수 없는 유형이라 CV 0.98 근처가 상한에 가깝다고 본다 [추론]. 추가 개선은 소폭일 것으로 예상한다.
- 최종 CV 점수는 후보 선택에 쓴 점수라 약간 낙관적이다. 독립 성능 추정치로 보고하지 않는다.
