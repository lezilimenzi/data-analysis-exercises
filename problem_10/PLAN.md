# PLAN — 문제 10 (영문 보고 5분야 다중 라벨 분류, Macro-F1)

근거는 `RESEARCH.md` 참고. 작성일 2026-10-10. 2026-10-10 검토 결과 반영본: 임베딩 제외, 프로토콜 기준 A, test 정보(라벨 없음) 사용 허용, 3단계 하이퍼파라미터 조정 추가(§5-1).

## 0. 전략 한 줄 요약
가장 약한 **logistics**(양성 84개, 임계값 0.5에서 F1 0.05)와 **shelter**를 세 가지로 보완한다.
1. valid를 학습에 합쳐 양성 수를 늘린다(logistics 48→84).
2. 분야별 임계값을 그룹 CV의 OOF로 고른다.
3. 문자 n-gram, NB 가중, 같은 문서 평활화를 후보로 둔다.

모델은 희소 선형 모델(TF-IDF + LogisticRegression)로 고정하고 하이퍼파라미터는 탐색하지 않는다. 모든 선택은 **문서 단위 StratifiedGroupKFold**의 짝지은 비교로만 한다.

| 약점 (RESEARCH §) | 대응 | 과적합 방지 장치 |
|---|---|---|
| logistics 임계값 붕괴 (§7) | 분야별 임계값 T | OOF를 여러 fold 모아서 고름, 0.05 간격 거친 격자, 동률이면 0.5에 가까운 값 |
| logistics 학습 양성 부족 (§4) | 프로토콜 B: fit+valid 합쳐 학습 | 프로토콜 A(기존 분할)와 같은 짝으로 비교 |
| 어휘 변형·희귀 단어 (§6) | char_wb(2–5) n-gram | 하이퍼파라미터 고정 |
| 짧은 텍스트, 키워드 의존 (§6, Wang & Manning) | NB 로그 비율 가중 LR | NB 비율은 학습 fold 라벨로만 계산 |
| 문서 안 라벨 순도 높음 (§5) | 같은 문서 확률 평활화 | α=0.5 고정, 라벨 미사용 |
| 문서 상관으로 인한 낙관적 CV (§5) | 그룹 분할 | groups=document_id |

## 1. 산출물과 파일
- 새로 만들 파일: `problem_10/problem.ipynb`, 실행 결과 `problem_10/submission.csv`.
- 실험 기록 `problem_10/EXPERIMENTS.md`: 요청 시에만 작성.
- 수정할 기존 파일: 없음.

## 2. 공통 설정 (고정)
```python
L = ['food','health','shelter','wash','logistics']
# 읽기: encoding='utf-8-sig'
# 벡터화: train+test 전체 텍스트에 1회 적합 (라벨 미사용)
W = TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True)
C = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,5), min_df=2,
                    sublinear_tf=True, max_features=200000)
LR = LogisticRegression(C=4, class_weight='balanced', max_iter=2000)   # 분야별 1개
# 마스킹: 분야 l의 학습·평가·임계값 선택에서 y_l == -1 행 제외
GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)   # 임계값 격자
```
- **임계값 선택 T(p, y)**: GRID 중 F1이 최대인 값을 고른다. 동률이면 0.5에 가장 가까운 값, 그것도 같으면 작은 값을 고른다.
- **NB-LR**: 학습 행 기준으로 r = log((Σ_{y=1}X + 1)/‖·‖₁) − log((Σ_{y=0}X + 1)/‖·‖₁)를 계산하고, X·diag(r)로 LR을 학습한다(Wang & Manning 방식, LR 버전). r은 분야마다, fold마다 따로 계산한다.
- **문서 평활화 S**: 같은 예측 집합(평가 fold 또는 test) 안에서 p'_i = 0.5·p_i + 0.5·mean_{j∈doc(i)} p_j를 쓴다. 자기 자신도 평균에 포함하므로 1행 문서는 그대로다. GroupKFold라 문서가 fold 안에 통째로 있다.

## 3. 평가 짝 (결과 보기 전 고정)
- train 7500행(fit+valid)을 `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=s)`로 나눈다. s ∈ {0, 1}이므로 **짝은 10개**다.
- groups=document_id, 층화 기준 y = (logistics == 1). 각 fold에 logistics 양성이 들어가게 하기 위함이다.
- **짝 점수**: fold k 안에서 분야별로 y≠-1인 행만 F1(float64)을 계산하고 5개를 평균한다. 분모가 0이면 그 분야 F1=0으로 두고 횟수를 기록한다.
- **fold k의 임계값**: 같은 seed에서 k가 아닌 4개 fold의 OOF 예측을 모은 표본으로 T를 고른다(fold별 평균이 아닌 모은 표본, Forman & Scholz).
- d_i = 후보 − 기준. |d_i| ≤ 1e-9이면 동점으로 센다. 보고는 CLAUDE.md §5를 따른다(전체 짝 수, 개선/악화/동점 수, 평균 d).

## 4. 1단계 — 모델 후보 (프로토콜 B로 학습: fold k 밖의 모든 행)
기준 **B0** = word TF-IDF + LR, 임계값 0.5.

| 등록 순서(=비용 순위) | 후보 | 내용 |
|---|---|---|
| C1 | word LR + T | 임계값만 조정 |
| C2 | word+char LR + T | 문자 n-gram 추가 |
| C3 | word+char NB-LR + T | NB 가중 |
| C4 | C2 + S + T | 문서 평활화(T는 평활화된 OOF로 선택) |
| C5 | C3 + S + T | NB + 평활화 조합(사전 등록) |

- 채택 조건: 평균 d > 0 **그리고** 개선 짝 수 > 악화 짝 수.
- 통과한 후보 중 평균 d가 최대인 것을 고른다. 최고값과 1e-9 이내인 후보끼리는 등록 순서가 빠른 것을 고른다. 모두 탈락하면 B0를 쓴다. 고른 모델을 **S\***라 부른다.
- 참고 보고(선택에는 쓰지 않음): C2−C1(문자 효과), C3−C2(NB 효과), C4−C2(평활화 효과)의 짝별 d, 분야별 OOF F1, 고른 임계값.
- **제외 후보 E (MiniLM 임베딩 + LR)**: 계산 예산 규칙으로 결과를 보기 전에 제외한다.
  - 예산은 임베딩 단계 3분 이하로 둔다. 실측은 318.6초였고 재측정 추정은 348–452초였다.
  - torch와 sentence-transformers가 requirements.txt 밖 의존성이라는 점도 제외 이유다.
- **제외 후보 (사유)**:
  - classifier chain: 분야 간 상관이 약하고, 오류가 전파될 위험이 있다.
  - 사전확률 보정: fit과 valid의 양성률이 같다.
  - C·n-gram 등 하이퍼파라미터 탐색: 선택 과적합 위험이 있다(Cawley & Talbot).

## 5. 2단계 — 학습 데이터 프로토콜 비교 (S\* 고정 후, 같은 10개 짝)
| | 학습 | 임계값 | 최종 제출 방식 |
|---|---|---|---|
| **A (기준)** | fold k 밖의 **fit 행만** | fold k 밖의 **valid 행**에 대한 그 모델의 예측으로 T 선택(깨끗한 holdout) | fit 6000행으로 학습, valid 1500행으로 T 선택 |
| **B (후보)** | fold k 밖의 **모든 행** | 1단계와 같음(다른 fold의 OOF) | 7500행 전부로 학습, 전체 OOF(2 seed 합침)로 T 선택 |

- S\*가 평활화를 쓰면 A의 임계값 선택에도 평활화된 예측을 쓴다.
- 채택 규칙은 §3과 같다. B가 통과하면 B, 아니면 A를 쓴다.
- **알려진 근사**: B에서 fold k의 임계값을 고르는 OOF(fold j의 예측)는 fold k 라벨로도 학습된 모델이 만든 것이다. 임계값 선택 자체는 fold k 라벨을 쓰지 않는다. 문제 9에서 허용한 방식과 같다.

## 5-1. 3단계 — 하이퍼파라미터 조정 (2026-10-10 검토에서 추가, 결과 보기 전 등록)
- 실행 조건(사전 고정): 2단계가 끝난 시점이 노트북 시작 후 **15분 이하**일 때만 실행한다. 넘으면 건너뛰고 이유를 출력한다.
- 기준은 1·2단계에서 고른 S\*와 프로토콜(C=4)이다. 후보는 LR의 **C=1, C=16** 두 개(등록 순서 C=1 → C=16)이고, 나머지 설정은 그대로 둔다.
- 짝, 판정 규칙, 동점 처리는 §3과 같다. 평가는 2단계에서 고른 프로토콜 방식으로 한다.
- 트랜스포머 미세조정은 CPU에서 MiniLM 순전파만으로 5분 이상 걸려 제외한다.

## 5-2. 4단계 — 오류 분석 기반 보완 후보 (2026-10-10 사후 추가, 결과 보기 전 등록)
근거: EXPERIMENTS.md §8. 1–3단계 결정(C2, 프로토콜 B, C=4)은 고정. 노트북 `problem_v2.ipynb` → `submission_v2.csv` (기존 `submission.csv` 보존).
- **짝**: 새 seed **{2, 3}** × StratifiedGroupKFold 5 = 10개(가설을 seed 0·1에서 만들었으므로). 층화·그룹·짝 점수·fold 임계값 규칙은 §3과 같음. seed별 pooled Macro-F1도 보고.
- **기준**: C2 (word+char TF-IDF LR, C=4, balanced, T).
- **후보** (등록 순서 = 비용 순위, 조합은 D4만 등록):

| 후보 | 정의 (고정) |
|---|---|
| D1 | 숫자 정규화: 두 벡터화기의 전처리 `re.sub(r'\d','0', s.lower())`, 나머지 C2와 같음 |
| D2 | 스태킹: 1단계 C2 OOF 확률 p(5) + 같은 문서 다른 행 평균 p(5, 1행 문서는 자기 p) + log(문서 크기) → 분야별 LR(C=1, balanced). 2단계는 같은 fold로 OOF를 만들고 T를 고름. 최종 2단계는 두 seed의 OOF 특징을 합쳐 학습, test 특징은 전체 학습 C2의 test 확률로 계산 |
| D3 | `CalibratedClassifierCV(LinearSVC(C=1.0, class_weight='balanced'), method='sigmoid', cv=3)` on word+char |
| D4 | C2와 D3 확률의 단순 평균 + T |
| D5 | word+char + LSA: `TruncatedSVD(300, random_state=0)`를 train+test word+char 행렬에 적합, 행 L2 정규화 후 희소 행렬에 붙여 LR |

- **판정**: CLAUDE.md §5 (평균 d>0 그리고 개선>악화), 통과 중 평균 d 최대, 1e-9 이내 동점은 등록 순서. 모두 탈락하면 C2 유지.
- **최종**: 고른 모델을 7500행 전체로 학습, 임계값은 seed 2·3 OOF를 합친 표본에서 선택.
- **시간 규칙**: D5 시작 시점이 노트북 시작 후 20분을 넘으면 D5를 건너뛴다.
- **알려진 근사**: D2 2단계 입력은 1단계 OOF(평가 fold 라벨로 학습된 모델이 다른 fold를 예측한 값)를 쓰고, test에는 전체 학습 모델 확률을 씀(척도 차이 가능). D3의 내부 보정 cv=3은 그룹 분할이 아님.

## 5-3. 5단계 — 사후 추가 후보 (2026-10-10, test 채점(v1 0.6736, v2 0.6760) 이후 등록. **사후 변경**)
노트북 `problem_v3.ipynb` → `submission_v3.csv` (기존 제출 파일 보존). test 점수는 선택에 쓰지 않는다.
- **짝**: 새 seed **{4, 5}** × StratifiedGroupKFold 5 = 10개 (seed 2·3은 D5 선택에 사용). 나머지 규칙은 §3과 같음.
- **기준**: D5 (word+char TF-IDF + LSA300 → LR C=4 balanced, T).
- **후보** (등록 순서 = 비용 순위):

| 후보 | 정의 (고정) |
|---|---|
| E3 | D3+D5 조합: `CalibratedClassifierCV(LinearSVC(C=1.0, class_weight='balanced'), sigmoid, cv=3)` on D5 특징(word+char+LSA) |
| E1 | D5 특징 + MiniLM-L6 임베딩(384, `normalize_embeddings=True`, 4스레드, max_seq 256) 결합 → LR(C=4, balanced) |
| E2 | D5 확률과 [MiniLM 임베딩만 → LR(C=4, balanced)] 확률의 단순 평균 |

- **임베딩 비용 규칙**: 처음 300문장으로 시간을 재서 9000문장 예상이 10분을 넘으면 E1·E2를 제외.
- **판정**: CLAUDE.md §5. 통과 중 평균 d 최대, 1e-9 이내 동점은 등록 순서. 모두 탈락하면 D5 유지.
- **임계값 절차 변경 (항목 2, 짝 비교 없이 사전 고정)**: 최종 모델의 임계값은 seed {4,5,6,7,8} × 5 fold OOF를 합친 표본에서 고른다. seed별로 고른 임계값도 진단으로 출력.
- **최종**: 고른 모델을 7500행 전체로 학습.

## 6. 노트북 구성과 시간 예산 (계산 시간, 측정값 기반 추정)
| 단계 | 작업 | 예상 |
|---|---|---|
| S0 | import, 시작 시각 기록, 데이터 로드, 열·순서 확인 | 5초 |
| S1 | word/char TF-IDF 적합 | 약 11초 |
| S2 | **안전 제출본**: B0를 7500행으로 학습, 임계값 0.5, `submission.csv` 저장 | 약 3초 |
| S3 | 1단계: 10짝 × (word LR, word+char LR, NB-LR) OOF + 평활화 + T | 약 1.5–2분 |
| S4 | 1단계 판정표 출력, S\* 결정 | – |
| S5 | 2단계: 10짝 × A 모델 | 약 40초 |
| S6 | 2단계 판정, 최종 학습·임계값·예측 | 약 10초 |
| S7 | 제출 검증 후 `submission.csv` 덮어쓰기 | – |
| | **합계** | **약 3–4분** |

- 시간 차단 규칙(사전 고정): S3 시작 후 10분이 지나도 끝나지 않으면 C3과 C5(NB)를 계산에서 빼고, 그 이유를 출력한다.

## 7. 검증
- 제출 파일: shape (1500, 6), 열 = sample_submission 열, `message_id` 순서가 sample과 같음, 값은 {0,1} 정수, 결측 없음.
- 정상성 확인: test 분야별 예측 양성률을 출력하고 train 양성률(§4)과 비교한다(판단 근거가 아니라 이상 탐지용).
- 재현성: random_state를 모두 고정하고 실행 결과를 노트북에 저장한다.
- 고른 모델의 CV 점수는 "선택에 쓴 점수"라고 표기하며, 독립적인 성능 확증으로 보고하지 않는다.

## 8. 가정
- test는 valid와 같은 방식(문서 분리, 전부 검토)이다. 분야별 F1은 test 전체 행에서 계산된다.
- train+test 텍스트로 TF-IDF 어휘와 IDF를 적합하는 것, test의 document_id로 평활화하는 것은 라벨을 쓰지 않으므로 허용된다.
- 외부 데이터(HumSet 등)와 사전학습 임베딩은 쓰지 않는다.
- OOF는 학습 데이터의 80%로 만든 모델에서 나오지만, 그 임계값을 100% 학습 모델에도 그대로 쓴다(확률 척도가 약간 달라질 위험이 있다).

## 9. 트레이드오프
- 선형 TF-IDF는 의미 일반화가 약하다. 대신 학습이 수 초라 짝 비교를 넉넉히 할 수 있고 실행 시간 위험이 거의 없다.
- 짝이 10개뿐이고 fold당 logistics 양성이 약 17개라 짝 점수의 분산이 크다. 그래서 후보를 5개로 제한했다.
- D00000(698행)이 한 fold에 통째로 들어가 그 fold의 분포가 다른 fold와 다르다. 그룹 분할을 지키는 대가다.
