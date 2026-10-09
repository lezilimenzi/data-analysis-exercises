# PLAN.md — 문제 5: 위성 이미지 토지피복 5클래스 분류

> 작성일: 2026-10-01 (개정 1: 사용자 답변 반영) · 상태: **검토 대기. Do not implement yet.**
> 근거: `RESEARCH.md`. 이전 문제의 설정값은 가져오지 않고 접근만 가져온다(`../STRATEGY.md`).
> 🔶 표시는 사용자 결정이 필요한 항목이다.

## 0. 사용자 결정 (2026-10-01)
| 항목 | 결정 |
|---|---|
| onnxruntime 설치 | 허용 → **설치 완료** (1.23.2, `--user`) |
| 지표 | Macro-F1 |
| 외부 사전학습 가중치·fine-tuning | **허용** → 3단계로 추가 |
| 시간 제한 | EDA부터 최종 배포까지 30분 |

## 1. 목표와 성공 기준
- train 750장으로 학습해 test 250장의 label(0~4)을 예측한다.
- 지표: `f1_score(y, p, labels=[0,1,2,3,4], average="macro", zero_division=0)`
- 산출물
  - `problem.ipynb`: EDA부터 제출 파일 생성까지 한 번에 실행되고, 출력이 남은 상태로 저장
  - `submission.csv`: `image_id,label`, 250행, test.csv 순서, `utf-8-sig`(sample과 같이 BOM 포함)
- 성공 기준
  1. 노트북 전체 실행 시간이 예산 안이다(§8).
  2. 형식 검사 통과(§7 셀 10).
  3. 최종 모델의 CV Macro-F1이 색 통계 기준선(0.778)보다 확실히 높다.
  4. 각 단계의 전체 Macro-F1과 하천 F1 변화를 표로 남긴다.

## 2. 환경 조사 결과와 추가 설치 🔶
[사실] 2026-10-01 확인.

| 항목 | 상태 | 조치 |
|---|---|---|
| Pillow | **9.0.1**. `feature_helper.py`가 쓰는 `Image.Resampling`이 없어 **helper가 실행되지 않음**. requirements.txt는 `Pillow>=10.0` | 🔶 `pip install --user "pillow==10.4.0"` |
| torchvision | 미설치. torch 2.10.0에 맞는 버전은 **0.25.0**(wheel 메타데이터 `torch==2.10.0` 확인) | 🔶 `pip install --user --no-deps torchvision==0.25.0` (`--no-deps`로 torch를 건드리지 않음) |
| 사전학습 가중치 | `download.pytorch.org`의 resnet18 가중치 URL 응답 200 | 첫 실행 때 `~/.cache/torch/hub/checkpoints/`에 자동 다운로드(약 45MB) |
| Jupyter | nbconvert 7.17.1, ipykernel 있음 | 노트북은 `jupyter nbconvert --execute --inplace`로 실행 |
| matplotlib | 3.5.1 (시스템). Pillow 10에서 문제가 생길 수 있음 | 노트북에서 쓰지 않는다. 몽타주는 PIL로 만들어 `IPython.display`로 표시 |

- 위험: Pillow를 `--user`로 올리면 이 계정의 다른 파이썬 작업에도 적용된다. 되돌릴 때는 `pip uninstall --user pillow`.

## 3. 측정한 실행 시간 [사실, CPU 8코어]
| 작업 | 측정값 | 환산 |
|---|---|---|
| ONNX 인코더 추론 (224, 스레드 2) | 5.1 ms/장 | 원본 1,000장 약 5초, 8방향 8,000장 약 40초 (+이미지 입출력) |
| ResNet18급 CNN 학습, 입력 64 | 7.6 ms/장 | 750장 1 epoch 약 6초 |
| 같은 모델, 입력 128 | 16.9 ms/장 | 750장 1 epoch 약 13초 |
| 같은 모델, 입력 224 | 78.2 ms/장 | 750장 1 epoch 약 59초 → **예산 초과, 쓰지 않음** |

- 학습 시간은 직접 만든 ResNet18 비슷한 구조로 잰 **근사치**다. 실제 torchvision 모델과 다를 수 있으므로 3단계 첫 epoch에서 다시 재고 §4 3단계의 후퇴 규칙을 적용한다.

## 4. 검증 설계
- **특징 기반 단계(1·2·4·5)**: `RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)` → 25개 fold. 같은 분할 객체를 계속 쓴다.
  - C 같은 값은 바깥 fold의 학습 부분 안에서 `GridSearchCV(cv=3)`로 고른다(중첩 CV).
- **fine-tuning 단계(3)**: 시간 때문에 `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)` 1회. 비교 상대(특징 기반 최선 모델)도 **같은 3개 fold에서 다시 평가**해 짝지어 비교한다.
  - 한계: 짝이 3개뿐이라 작은 차이는 판정할 수 없다. 큰 차이만 잡아낸다.
- 집계
  - 채택 판정: fold별 Macro-F1의 짝지은 차이.
  - 보고: pooled Macro-F1(반복별) 평균 ± std, 클래스별 F1, pooled 혼동 행렬.
- 전처리(스케일러 등)는 Pipeline 안에서 fold 학습 부분에만 fit한다.

### 4.1 사전 채택 규칙 🔶
| 비교 | 규칙 (모두 만족해야 채택. 아니면 단순한 쪽 유지) |
|---|---|
| 특징 기반 (25쌍) | 짝지은 차이 평균 ≥ T, 25쌍 중 15쌍 이상 개선 |
| fine-tuning 대 특징 최선 (3쌍) | 짝지은 차이 평균 ≥ T, **3쌍 모두** 개선 |

- T = max(0.005, s). s는 1단계 채택 모델의 반복별 pooled Macro-F1 표준편차. 1단계가 끝나면 출력하고 고정한다.
- 1단계 안의 비교(B1 대 B2)에는 잠정값 T = 0.005를 쓴다.

## 5. 단계별 실험 (조건부 순차)

### 0단계. 준비와 EDA
- RESEARCH.md의 EDA 중 핵심만 노트북에 넣는다: 행 수, 결측·중복 ID, 이미지 크기·모드, 라벨 분포, MD5 중복, 클래스별 색 통계 표.
- 원본 1,000장 특징 추출 → `cache/feat_orig_{train,test}.npz` (helper의 `cache_path`).

### 1단계. CNN 특징 기준선 (전략 A) — 무조건
| 후보 | 모델 | 내부 탐색 격자 |
|---|---|---|
| B0 | 색 통계 7개 + LR (참고, 재현 확인) | 없음 |
| B1 | 1280d + `StandardScaler` + `LogisticRegression(max_iter=5000)` | C ∈ {0.01, 0.1, 1} |
| B2 | 1280d + `StandardScaler` + `SVC(kernel="rbf")` | C ∈ {1, 10}, gamma="scale" |

- B1 대 B2를 §4.1(잠정 T)로 비교한다. 규칙을 못 넘으면 **B1(더 단순)**.
- **안전 제출본**: B1을 train 전체로 학습 → `submission.csv` 저장 → 형식 검사.
- 진단(출력만): 클래스별 F1, 혼동 행렬, 하천 대 목초지·하천 대 초본 이진 AUC(STRATEGY 6.4), 하천 오분류 몽타주(최대 24장).
- T 고정.

### 2단계. 8방향 회전·뒤집기 특징 평균 (전략 B) — 무조건
```python
OPS = [None, Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM,
       Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270,
       Image.Transpose.TRANSPOSE, Image.Transpose.TRANSVERSE]
# None: 원본 jpg 경로 그대로. 나머지 7개: RGB로 읽어 transpose → cache/views/{k}/{image_id}.png (무손실)
feat_d8 = np.mean([extract_features(paths_k, cache_path=...) for k in range(8)], axis=0)
```
- 후보 S2 = 1단계 채택 모델 + `feat_d8`(입력만 교체, 격자 동일).
- 평균 방식이라 행 수가 늘지 않고, 같은 이미지의 변형이 fold를 넘나드는 누수가 없다. 8배 증강 학습은 하지 않는다.

### 3단계. ResNet18 fine-tuning (전략 E) — 무조건
- 모델: `torchvision.models.resnet18(weights="IMAGENET1K_V1")`, `fc`를 5클래스로 교체. 전체 층 학습.
  - ResNet18을 고른 이유: 측정한 CPU 학습 시간 안에 들어오고 구조가 표준적이다. 다른 백본 비교는 시간 때문에 하지 않는다.
- 입력: 64×64 → **128×128** bilinear 업샘플 → ImageNet 정규화.
- 증강: 학습 때 무작위 8방향(뒤집기 + 90° 회전)만. 색 변환은 하지 않는다(색이 클래스 정보다).
- 학습 설정(관행값으로 고정, 이 데이터에서 비교하지 않음):
  - AdamW, lr 3e-4, weight decay 1e-4, cosine 스케줄, **12 epoch**, batch 32
  - `torch.manual_seed(42)`, `torch.set_num_threads(8)`
- 예측: 8방향 TTA(softmax 평균).
- 평가: §4의 3-fold. fold마다 epoch별 검증 Macro-F1을 기록한다(진단용, epoch 선택에는 쓰지 않는다).
- 비교: 1·2단계 채택 특징 모델을 같은 3-fold에서 평가 → §4.1 규칙.
- **시간 후퇴 규칙(사전 고정)**: 첫 fold 첫 epoch 시간 × (3 fold × 12 + 최종 12 × 1.5)가 **12분을 넘으면** 입력을 64×64로 바꾸고 처음부터 다시 시작한다. 64로도 넘으면 3단계를 중단하고 특징 모델을 유지한다.
- 예상 시간(128 기준): fold당 학습 500장 × 12 epoch ≈ 1.7분 → 3 fold ≈ 5분 + 최종 750장 12 epoch ≈ 2.5분 → **약 8분**.

### 4단계. 수면·선형 구조 수작업 특징 (전략 C) — 조건부
- **실행 조건**: 3단계에서 특징 모델이 유지됐고, 하천 F1이 가장 낮으며, 남은 예산이 있을 때.
- 정의(라벨 미사용, 고정):
  ```python
  rgb = img / 255.0;  R, G, B = rgb[...,0], rgb[...,1], rgb[...,2]
  w = B / (R + G + B + 1e-6) - rgb.mean(-1)    # 푸르고 어두울수록 큼
  mask = w >= np.quantile(w, 0.85)             # 이미지별 상위 15%
  big = scipy.ndimage.label(mask)의 가장 큰 연결 성분
  feats = [big.sum()/mask.sum(), elongation(big), w[mask].mean()-w[~mask].mean(),
           *np.quantile(w, [0.1, 0.5, 0.9]), coherence(gray)]   # 7개
  ```
- 후보 S4 = 현재 최선 입력 + 7개(StandardScaler 후 연결). 효과가 없으면 추가 튜닝 없이 미채택.

### 5단계. 하천 결정 가중치 (전략 D) — 조건부
- **실행 조건**: 특징 기반 LR이 최종 후보이고, 하천이 가장 낮은 F1 클래스이며, 하천 recall < precision일 때. SVC나 fine-tuning이 채택되면 건너뛴다.
- w ∈ {1.0, 1.2, 1.4, 1.6, 2.0}을 바깥 fold 학습 부분의 내부 OOF에서만 고른다(STRATEGY 10.3 함수).

## 6. 결과 기록 형식 (노트북 안 출력)
| 단계 | 후보 | 평가 | Macro-F1 (평균 ± std) | 하천 F1 | Δ 대비 현재 최선 | 개선 쌍 수 | 채택 |
|---|---|---|---|---|---|---|---|

## 7. 노트북 셀 구성 (`problem.ipynb`)
1. import, 경로·seed 상수, 시작 시각 기록, CSV 로드(`utf-8-sig`)
2. EDA(0단계 항목)
3. 평가 함수, 공통 분할 객체, 결과 기록 헬퍼
4. 원본 특징 추출(캐시)
5. 1단계: B0·B1·B2, 안전 제출본, 진단, T 고정
6. 2단계: 8방향 PNG 생성 → 특징 → S2
7. 3단계: fine-tuning 3-fold + 특징 최선 모델 같은 fold 재평가 → 비교
8. 4·5단계: 조건 검사 → (충족 시) 실행
9. 최종 모델: 채택 구성으로 train 전체 학습 → test 예측 → `submission.csv`
10. 형식 검사(열, 250행, ID 집합·순서 일치, 정수, 0~4) + 예측 분포 출력 + 총 실행 시간 출력

각 셀 끝에 누적 경과 시간을 출력한다.

## 8. 시간 예산 🔶
- 해석: "EDA부터 최종 배포까지 30분"을 **노트북을 처음부터 끝까지 실행하는 시간 ≤ 30분**으로 본다. 코드 작성 시간은 포함하지 않는다.

| 구간 | 예상 |
|---|---|
| EDA + 원본 특징 | 1분 미만 |
| 1단계 (25 fold 중첩 CV) | 2~3분 |
| 2단계 (PNG 7,000장 + 추출 + CV) | 2~3분 |
| 3단계 (128 기준) | 약 8분 |
| 4·5단계 (조건부) | 각 1~2분 |
| 최종 예측·검사 | 1분 미만 |
| **합계** | **약 15~18분** (여유 약 12분) |

- 후퇴 순서: 5단계 → 4단계 → B2 → 3단계 입력 64로 축소. 1·2단계와 안전 제출본은 유지한다.

## 9. 생성·수정 파일
| 경로 | 내용 |
|---|---|
| `problem.ipynb` | 새로 생성 |
| `submission.csv` | 새로 생성. 안전 제출본 → 최종본으로 덮어쓴다 |
| `cache/` | 특징 npz, 8방향 PNG(약 7,000장) |
| `~/.cache/torch/hub/checkpoints/` | ResNet18 가중치(자동 다운로드) |
| 기존 파일 | 수정하지 않는다 (`feature_helper.py`, `data/`, `models/`) |

## 10. 검증 체크
- [ ] B0 재현: 색 통계 LR이 RESEARCH의 0.778 근처(분할이 달라 완전히 같지는 않다)
- [ ] 특징 shape (750,1280)/(250,1280), 유한값
- [ ] 8방향 중 identity 특징이 원본 특징과 같다
- [ ] 회전 PNG를 다시 읽어 역변환하면 원본과 픽셀이 완전히 같다
- [ ] 특징 단계는 모두 같은 25개 fold, 3단계 비교는 두 모델이 같은 3개 fold
- [ ] fine-tuning 검증 fold 이미지가 그 fold의 학습에 들어가지 않았다(인덱스 assert)
- [ ] 최종 제출 형식 검사 통과, 총 실행 시간 ≤ 30분

## 11. 하지 않는 것
- 혼동 쌍 전용 이진 분류기 재분류(P2에서 악화).
- fine-tuning 모델과 특징 모델의 앙상블, 시드 앙상블, 다른 백본 비교(시간 예산).
- fine-tuning 학습률·epoch 탐색(시간 예산. 관행값 고정의 한계로 기록).
- test 예측 분포를 보고 결정 규칙을 바꾸는 것.
- 외부 데이터·EuroSAT 원본 사용.
