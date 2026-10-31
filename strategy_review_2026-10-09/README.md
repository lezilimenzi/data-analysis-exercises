# P1~P8 전략 보완 검토 — 2026-10-09

대상 문서: [STRATEGY_full_2026-10-05.md](../STRATEGY_full_2026-10-05.md).
수험용 입력·문제 설명·연구/실험 기록·코드·저장 OOF를 대조했다.
instructor 자료, test 정답, 채점기는 사용하지 않았고 기존 제출과 모델 코드를 변경하지 않았다.

**후속 검증**: 아래는 첫 검토 시점의 결과다. 사용자의 추가 요청으로 P1 최종 앙상블과
P2~P7의 미검증 후보를 실제 비교한 기록은 [후속 결과](../strategy_validation_2026-10-09/README.md)와
[후속 프로토콜](../strategy_validation_2026-10-09/PROTOCOL.md)을 따른다. 아래의 "미검증"은 첫 검토 당시의 상태다.

## 이번에 확인한 핵심

| 항목 | 결과 | 근거 수준 |
|---|---|---|
| P1 결측 대체 후 도메인 파생 계산 | RF 양성 F1 0.870394 → 0.873719, +0.003325, 10/0/0 | 새 학습. 최초 3회 선별 후 7회 추가 분할; 같은 train 재사용 |
| P1 ET·RF 전체 특징 후보 | 최초 3회에서 각각 −0.083933 / −0.002774 | 새 학습, 미통과 |
| P5 정규화 변경 3후보 | 모두 기준보다 평균 낮음, 채택 규칙 미통과 | 새 학습, 15개 바깥 fold·각각 내부 규제 선택 |
| P5 누락된 사후 앙상블 | LR 대비 +0.018603, 5/0/0; FT 단독 대비 +0.010688, 4/1/0 | 기존 저장 OOF에서 재계산. fine-tuning 허용 별도 조건 |
| P6 단서 특징·고정밀 단서 선택 | 새 오류가 더 많아 미통과 | 기존 Run 11~12 및 CSV 확인; 재학습 아님 |
| P8 시점 이동 | fit 경계→valid 원 점수 1.0; −3샘플 이동 시 0.997659 | 합성 스트레스. 실제 운전 이동 확률이나 test 점수 아님 |

P1의 짝은 반복별 pooled 양성 F1이다. P5는 fold별 고정 5클래스 Macro-F1이다.
SD는 짝 차이의 모집단 표준편차(`ddof=0`)로 원 전략 문서 방식과 맞췄다.
수치 동점 허용오차 1e−9, 평균 d>0이고 개선 수>악화 수를 운영상 통과로 판단했다.
P1 원 RF 점수는 기존 10회 모두를 절대 오차 1e−12 이내로 재현했다.
P1 대체 순서 개선이 원 최종 RF·GB 앙상블 전체에서도 유지되는지는 확인하지 못했다.

P2·P3·P4·P6·P7·P8의 추가 제안은 문서에 미검증으로 구분했다.
P2 저장 CNN OOF 오류 재집계, P3 타깃 경계 검사, P7 단일 클래스 지표 검사는
새 모델 학습과 별개다. 새 독립 데이터의 성능 확증은 없다.

## 파일과 재현

- [PROTOCOL.md](PROTOCOL.md): 실행 전 후보·분할·채택 규칙, 1차 결과 뒤 추가한 P1 후속 점검 설계.
- [audit.py](audit.py) → [data_audit.json](data_audit.json): 8문제 입력 감사·P8 스트레스.
- [experiments.py](experiments.py): P1/P5 새 학습. `*_pairs.csv`는 반올림하지 않은 짝별 점수,
  `*_summary.json`은 요약, `*_oof_*.npz`는 train OOF 라벨이다. P5 내부 C는 `p5_inner_choices.csv`.
- [recheck.py](recheck.py) → [recheck.json](recheck.json): 과거 P2/P5 저장 확률 재집계,
  P3 경계·P7 지표 검사, P1 최초/후속 합계와 고침/망침 진단.
- `.log`: 실제 실행 출력. 초기 P1 스크립트의 `predict`는 원 실험의 `p>=0.5`와
  정확히 같은 규칙이 아니어서 결과 해석 전에 중단하고 수정·재실행했다.
  현재 로그·CSV·OOF는 수정된 규칙의 완주 결과다. 후보나 채택 규칙 변경은 아니다.

프로젝트 루트에서:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 strategy_review_2026-10-09/audit.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 strategy_review_2026-10-09/experiments.py p1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 strategy_review_2026-10-09/experiments.py p1_confirm
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 strategy_review_2026-10-09/experiments.py p5
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 strategy_review_2026-10-09/recheck.py
```

재실행하면 이 검토 폴더 안의 결과를 덮어쓴다. 원 데이터·제출은 수정하지 않는다.
P5 캐시는 모델·전처리 식별자·순서 있는 이미지 내용 해시가 일치할 때만 사용한다.

## 실행 환경과 시간

Python 3.10.12, NumPy 1.26.4, pandas 2.3.2, SciPy 1.8.0, scikit-learn 1.7.2.
CPU 실행, GPU 없음. BLAS/OMP 스레드 1개, 트리는 작업 2개.
완주 로그 기준 P1 최초 약 148초, P1 후속 약 134초, P5 약 78초.
병행 실행·장비 차이의 영향을 받으므로 시험 환경 비용을 보장하지 않는다.

설치된 SciPy가 선언한 NumPy 버전 범위를 벗어난다는 경고가 있었다.
이번 실행은 완료됐고 P1 기존 기준 점수는 재현했지만 환경 호환성이 검증됐다는 뜻은 아니다.
패키지를 변경하지 않았으며, 향후 재현 환경에서는 호환 버전을 맞춘 뒤 기준 점수를 다시 확인한다.

## 원문·평가의 한계

P1~P3 `problem.md`, P5 Codex `problem.md`는 직접 확인했다.
P4·P6·P7·P8 노트북에는 원 지문이 남아 있지 않아 연구 기록과 실행 코드로 과제를 대조했다.
이들 규칙을 원문으로 다시 확인하지 못했다. 기존 외부 문헌 전체를 재검토한 것도 아니다.
이번에 공식 문서로 확인한 것은 F1 집계, Normalizer, RF 특징 수, 전처리 누수 원칙이다.

문서에서 바로잡은 주요 해석은 pooled F1 ≠ 그룹 F1 가중평균,
단일 클래스 세션의 고정 라벨 지표, 점수 동점 ≠ 예측 일치,
P8 요약 간격만으로 정보 손실을 단정할 수 없음,
전체 라벨을 본 EDA가 외부 평가의 독립성을 제한한다는 점이다.
