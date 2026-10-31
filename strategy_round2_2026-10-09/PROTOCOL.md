# 추가 실험 2차 — 결과 확인 전 고정

사용자 요청: P1–P8에서 추가로 실험할 요소를 실제 확인한다.
기존 흐름을 재사용하는 제한된 연구 실험이다. 수험용 train만 사용하며 제출 파일은 변경하지 않는다.
여러 차례 살펴본 데이터이므로 독립 확증 또는 공식 점수 향상을 주장하지 않는다.

## 후보와 비교 조건

| 문제 | 기준 | 고정 후보 | 평가 단위 |
|---|---|---|---|
| P1 | raw median 대체 후 파생, RF300+GB 1:1, p≥.5 | raw StandardScaler→IterativeImputer(BayesianRidge 기본, max_iter=10, seed=7); 역변환 후 결측 칸만 대체하고 파생. 관측값 보존, imputer는 학습 fold만 fit | 기존 stratified 5-fold×10회, seed42..51; 반복별 pooled 양성 F1 |
| P2 | GAP CNN | 같은 seed/fold GAP와 grid의 softmax 1:1 평균, TTA 없음 | 저장된 3-fold×3seed 확률; 9짝 Macro-F1, 3seed ensemble |
| P3 | 앞선 실험의 M2 잔차 3점 평활화 | 같은 평활화 + 학습 origin의 나이에 반감기180일 가중. 가중치 평균1, StandardScaler와 Ridge(alpha1) 모두 가중 fit | 기존 2014-01-01~12-25 23시 359 origins, 72h MAE |
| P4 | stats36+S3 72, ET500 balanced seed42, class1×1.4 | 6채널 간 Pearson 상관 15개 추가. 15개 시점 내부만 사용, 분산0인 쌍은0 | 기존 stratified 5-fold×10회(seed0..9), fold Macro-F1 |
| P5 | 제공 encoder1280→StandardScaler→L2→LR | scaler 뒤 PCA128(randomized, seed7, whiten=False) 추가; 나머지 동일 | 기존 outer5fold×3(seed20261009..11), inner3fold(seed7) C[.01,.1,1,10,100], fold Macro-F1 |
| P6 | char_wb(2,5) TF-IDF + balanced LinearSVC C1 + inner threshold | char_wb(2,5)와 char(3,5) TF-IDF를 같은 비중으로 연결 후 L2. 두 블록 모두 min_df2, sublinear_tf | 기존 outer StratifiedGroupKFold5×3, inner StratifiedKFold3(seed7), 임계값41개; fold Macro-F1 |
| P7 | F7+지속성3, scaler→LR C1, p>.5 | 학습 행 가중치 n/(세션수×해당 세션 행수), scaler와 LR 모두 적용 | 기존 14-session LOSO, 세션 Macro-F1 평균; pooled 보조 |
| P8 | pressure>20 첫 시점, fit 라벨로 경계 | pressure>20이 3표본 연속인 구간의 시작 시점, fit으로 별도 경계 | 기존 clean fit1324/valid440; valid의 고정30%에 900..919 위치부터 25bar 1/2/3표본 충격 주입(seed20261009), 각 조건 Macro-F1 |

P6 char 단독은 과거에 탐색했으나 여기서는 두 표현의 고정 결합을 확인한다. 조합 외 추가 튜닝 없음.
P8 주입 위치는 라벨·개별 상승 시점에 의존하지 않는 고정 구간이다. clean fit만 경계 학습에 사용하며
3표본 잡음까지 포함해 후보의 실패 경계도 측정한다. 합성 조건의 견고성이지 실제 잡음 빈도나 실전 점수의 증거는 아니다.

## 판정·비용·검증

- P1/P3/P4/P5/P6/P7: d=후보−기준(F1), 기준−후보(MAE). float64, |d|≤1e−9 동점.
  평균 d>0 AND 개선짝>악화짝이면 운영 규칙 통과. 통계적 유의성 또는 제출 교체를 의미하지 않는다.
  P1은 이전 엄격 기준 8/10 개선 여부도 별도 보고한다.
- P2: 이전 규칙 그대로 평균 d≥.003, 개선≥6/9, 3seed ensemble≥기준 모두 필요.
- P8: clean 유지 및 1/2표본 충격 개선 여부를 확인. 3표본 충격은 한계 시험이며 평균 내 채택하지 않는다.
- 저장된 기준 OOF는 라벨·분할 일치를 확인하고 사용한다. P3/P7은 기준 재계산도 확인한다.
- 결과를 본 뒤 후보·가중치·평가 범위를 바꾸지 않는다. 추가 조합은 실행하지 않는다.
- 순서 P2/P3/P7/P8→P4/P5/P6→P1. CPU, BLAS1, OMP1; CNN 재학습 없음.
- OOF/분할/선택값/점수/소요시간 저장 후 별도 스크립트에서 점수와 판정 재계산.
  P3 시간 경계, P6 외부 그룹 분리, P7 세션 분리 검증. 기준·코드·입력 SHA256 기록.
- 반복CV는 같은 행을 재사용하고 P3 72시간 목표도 겹친다. 짝을 독립표본으로 간주하지 않는다.
