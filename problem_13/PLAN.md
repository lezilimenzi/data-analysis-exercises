# PLAN — 문제 13 (화성 cone 탐지, AP50:95) 문제해결 전략

작성일 2026-10-13, 갱신 2026-10-14(CPU 벤치마크, A1 크레이터 근거, Ultralytics 필터 조건 정정). 근거: `analysis/EDA.md`, `../STRATEGY.md`(특히 2장·4장, P11 기록), 아래 레퍼런스. `problem_13_instructor`는 열지 않음.
표기: **[사실]** 데이터·코드에서 확인 / **[추론]** 검증 안 됨 / **[문헌 N]** 9절 번호.
상태: **실행 완료(2026-10-16). 결과는 `EXPERIMENTS.md`.**

## 1. 규칙 요약
| 항목 | 내용 |
|---|---|
| 지표 | 100 × AP50:95, 단일 클래스, 101점 보간, 전역 신뢰도 순위, 이미지당 최대 100개. `evaluation.evaluate`가 기준 |
| 제약 | CPU 전용(`device='cpu'`), 시간 제한 없음(학습 시간은 기록). 원문은 Notebook에 학습·추론 코드와 출력을 저장하라고 하지만, **2026-10-14 사용자 결정으로 Notebook 없이 `.py` 스크립트 + 로그로 진행** |
| 허용 | 다른 모델·사전학습 가중치·미세 조정. **평가 영상을 학습한 외부 모델은 금지**, 원본 정답 검색 금지 |
| 분할 | 제공 fit/valid 그대로 사용. 설정은 valid로 선택. 이후 fit+valid 재학습 허용(이때 valid 점수는 독립 검증이 아님) |
| 금지 특징 | image_id·scene_id를 예측 특징으로 쓰지 않음 |
| 제출 | `image_id,class_id,confidence,x_min,y_min,x_max,y_max`, 검출 없는 이미지는 행 생략, `index=False` |

## 2. EDA에서 가져온 전략 근거 (사실 → 전략)
| EDA 사실 | 전략 결정 |
|---|---|
| test = 학습에 없는 scene 2개(G06·G07). 픽셀 통계는 G04·G05와 거의 같음 | valid(G05) 점수를 test의 대리 지표로 사용(추론). 새 scene에 대한 일반화가 목표 |
| 박스 72~77%가 타일 경계에서 잘림. 이웃 타일에서 99.6%가 이어짐 → 라벨은 마스크를 타일에서 자른 것 | 잘린 박스를 **지우지 않음**(P11과 반대). 모델이 보이는 부분만 박스로 잡도록 학습. mosaic·이동 증강도 경계에서 박스를 자르므로 유지. 단, 면적이 10% 이하로 남은 조각은 버린다는 점에서 GT 규칙과 완전히 같지는 않음 |
| GT 쌍의 최대 IoU 0.21 | NMS IoU를 기본값 0.7보다 낮추는 후보를 둠 |
| 모든 scene에서 조명이 왼쪽에서 옴. 위아래 방향은 scene마다 다름 | 기본 `fliplr=0.5`가 물리적으로 없는 영상을 만듦 → **fliplr=0, flipud=0.5** 후보. Ultralytics TTA(`augment=True`)도 좌우 반전을 쓰므로 그대로 쓰지 않음 [사실: `tasks.py` `_predict_augment`] |
| 계조 55~70단계, p1~p99가 114~141 DN. G01은 대비가 2배 | 대비 정규화(percentile 스트레치 또는 CLAHE) 후보 [문헌 9: 저대비에서만 이득, 검증 필요] |
| 박스 크기 중앙값 99 px, small 9%. 16 px 박스는 2 px 오차에 IoU 0.62 | 해상도를 512보다 줄이지 않음. 640 업샘플은 비용 대비 후보 |
| nodata 타일 6.3%, nodata 경계에서도 박스가 잘림 | 예측 박스를 0이 아닌 영역으로 자르는 후처리 후보 |
| 변 ≤ 2 px GT 32개(1.1%). Ultralytics는 증강 뒤 w 또는 h가 2 px 이하이거나 면적이 원래의 10% 이하로 줄어든 박스를 버림(`w2>2 & h2>2 & 면적비>0.1`) → 확대 증강 때만 살아남음 [사실: `augment.py` `box_candidates`] | 이 조각들은 사실상 포기(재현율 상한 약 99%). 별도 처리하지 않음 |
| 배경 87%(fit). scene 양성률 3.7~91% | 배경 샘플링 비율을 후보로 둠(`prepare_yolo(negative_ratio)`) |

## 3. 검증 설계 (결과를 보기 전에 고정)
- **평가 세트**: valid 전체 2,209장. 학습 표본만 고르고 평가는 줄이지 않음(README).
- **대표 점수**: valid 전체에 대한 `evaluation.evaluate`의 `score_100`(pooled, 채점과 같은 방식). AP50, AP75, `recall_at_50`도 함께 기록.
- **짝 단위**: valid 이미지를 **고정 seed(13)로 무작위 10묶음**(각 약 221장)으로 나누고, 묶음별로 `evaluate`를 계산. 이 묶음 정의는 모든 후보에 똑같이 적용.
  - 한계: 같은 모델의 예측을 나눈 것이라 짝들이 독립이 아님. scene이 G05 하나뿐이라 scene 짝은 만들 수 없음. 운영 규칙으로만 사용(CLAUDE.md §5).
- **채택 규칙**: CLAUDE.md §5 그대로. d_i = 후보 − 기준(AP50:95, float64), |d| ≤ 1e-9이면 동점. 평균 d > 0 이고 개선 묶음 수 > 악화 묶음 수일 때 통과. 여러 후보가 통과하면 평균 d 최대, 동점이면 비용 순서·등록 순서로 고름.
- **보고**: 짝 수, 개선/악화/동점 수, 평균 d, pooled 점수. 고른 점수는 "선택에 쓴 점수"로 표시.
- **재현성**: 모든 학습에 `seed=0, deterministic=True`, `workers=0`. 학습마다 실행 시간과 스레드 수를 기록.
- **학습 중 검증 끔**: `val=False` → Ultralytics는 마지막 epoch에서만 한 번 평가함 [사실: `trainer.py` 649행]. 고정 E의 `last.pt`만 사용하고 `best.pt`는 쓰지 않음(valid로 체크포인트를 고르는 편향 방지). 후보 판정은 항상 `evaluation.py`로 다시 계산.
- **옵티마이저 명시**: `optimizer=auto`가 고르는 값(반복 1만 회 이하 → AdamW, lr0 0.002)을 버전과 무관하게 직접 지정.
- **학습 seed 재확인 규칙 (2026-10-14 사용자 결정으로 변경)**: 단계 중에는 seed 0 한 번으로 판정함. 마지막에 최종 설정(fit 학습)과 B0를 `seed=1`로 한 번씩 더 학습해 같은 10묶음으로 비교하고, 두 seed 모두에서 평균 d > 0인지 기록함. 한계: 중간 단계의 채택은 seed 운에 좌우될 수 있고, 마지막 재확인은 어느 변경이 효과였는지 가려내지 못함.

## 4. 단계별 실행 계획 (조건부 순차, P11 방식)
각 단계의 승자가 다음 단계의 기준이 됨. 단계 안의 후보는 같은 기준 대비 같은 짝으로 평가.

### 0단계 — 환경·비용 측정 → 기준선 저장
1. 환경: `uv`로 Python 3.12 venv를 만들고 requirements 고정 버전(ultralytics 8.3.221, torch 2.8.0 CPU)을 설치(사용자 확정). 현재 python3에는 ultralytics 8.4.175가 있음 [사실]. 설치에 실패하면 8.4.175를 쓰고 그 사실을 기록.
2. `prepare_yolo(negative_ratio=1.0)`(양성 903장 + 배경 903장)로 **1 epoch 시간을 측정**.
3. epoch 수 E는 측정 전에 정한 규칙으로 결정: `E = min(60, floor(60분 / epoch 시간))`, 최소 10 (2026-10-14 사용자 결정으로 예산 40분 → 60분). 이후 모든 후보는 같은 E를 사용(비교의 공정성).
4. **B0** = yolov8n COCO, imgsz 512, `single_cls`, 기본 증강 + `close_mosaic=2`(기본값 10이면 E≈10에서 mosaic이 아예 꺼짐 [사실: `trainer.py` 505행], 사용자 확정), AdamW lr0 0.002, batch 16, freeze 없음, E epoch, `negative_ratio=1.0`. 추론은 conf 0.001, NMS IoU 0.7, max_det 100. → valid 점수 기록 후 test 예측을 `submission.csv`로 바로 저장(강한 기준선을 먼저 확보).

### 1단계 — 입력 표현·증강 (재학습 필요, 후보 2개)
| ID | 변경 | 근거 |
|---|---|---|
| A1 | `fliplr=0, flipud=0.5` | 조명 방향 보존(EDA 3장, [문헌 7·8]). 같은 조명에서 솟은 지형(cone)과 파인 지형(크레이터)은 음영이 반대이므로, 좌우 반전한 cone은 크레이터와 비슷한 음영이 됨 → fliplr가 크레이터 오탐을 늘릴 수 있음 [추론, 기하는 사실] |
| C1 | 학습·추론 입력을 이미지별 p1~p99 스트레치로 정규화 | 낮은 계조, G01 대비 차이 |

- A1 판정은 위 규칙 그대로 함. 가설 확인용으로 B0·A1 각각 **배경 이미지의 오탐 수**(`detections_on_negative_images`, conf 0.25 이상인 수 포함)를 함께 기록. 판정에는 쓰지 않음.
- 진단(판정에 쓰지 않음): test scene(G06·G07)의 조명 방향을 라벨 없이 추정해 기록. 고대비 둥근 지형 크롭의 좌우 밝기 차이를 이용.

### 2단계 — 학습 자료·해상도 (후보 2개)
| ID | 변경 | 비용 |
|---|---|---|
| HN | 기준의 학습 목록(양성 903 + 무작위 배경 903)에 **hard negative 903장 추가**(총 약 2,709장) | epoch 시간 약 1.5배 |
| R640 | imgsz 640 (업샘플) | 약 1.6배 |

- **HN 정의(결과 전에 고정, 2026-10-14 사용자 결정으로 N3를 대체)**: 2단계 기준 모델(1단계 승자, seed 0)로 fit의 배경 이미지 중 기준 학습 목록에 없는 약 5,253장을 conf 0.001로 예측 → 이미지별 최고 신뢰도 순으로 상위 903장을 hard negative로 고름. 동점은 image_id 순. **fit만 사용하고 valid는 쓰지 않음.** seed 재확인 때도 같은 목록 사용. 903장(1배)은 비용을 위한 [판단].
- 근거: 배경이 fit 87%·valid 79%라 test도 배경이 대부분일 가능성이 높음 [추론] → 배경 이미지의 오탐이 AP를 깎음. 무작위 배경보다 모델이 실제로 헷갈리는 배경(크레이터·능선 등)이 정보가 많음 [추론].
- 위험: 상위 hard negative에 **라벨이 빠진 진짜 cone**이 섞여 있으면, 진짜 cone을 억제하도록 배우게 됨. 진단으로 상위 24장을 시각화해 기록만 하고, 라벨은 고치지 않음.
- 같은 E로 학습(반복 1.5배). 효과는 "hard negative + 추가 학습량"의 합친 효과로 기록.

### 3단계 — 모델 (후보 2개, COCO 가중치만)
| ID | 변경 | 근거 |
|---|---|---|
| M11 | yolo11n | COCO mAP 39.5 vs 37.3, CPU 지연시간이 더 짧음 [문헌 5] |
| ~~M8s~~ | ~~yolov8s~~ | **제외(2026-10-14, 결과 보기 전, 사용자 확정)**: 비용 약 3배(10 epoch 약 3.8시간)로 예산 초과 → 사전 조건에 따라 제외 |

- 다른 계열(RetinaNet·Faster R-CNN ResNet50-FPN)은 **비용 조건으로 제외**. CPU(4스레드)에서 512 px, 배치 4 학습 1스텝 시간을 쟀음: YOLOv8n 0.73초, RetinaNet 7.62초(약 10배), Faster R-CNN 10.77초(약 15배) [사실: `analysis/bench_cpu.py`, 무작위 가중치, 데이터 로딩 제외, YOLO는 대용 손실]. 배경 1:1 기준 1 epoch은 각각 약 5.5분, 57분, 81분으로 추정 → 학습 1회 60분 예산 안에서 RetinaNet은 1 epoch 정도만 가능. 백본을 고정한 RetinaNet은 시간을 재지 않아 미등록.
- Mars 사전학습 모델(MOMO, Mars-Bench 가중치)은 **제외**. MOMO는 Murray Lab CTX 전역 모자이크(test scene과 같은 출처)로 사전학습되었고 [문헌 4], Mars-Bench의 conequest 가중치는 ConeQuest 라벨을 학습했을 수 있음 → "평가 영상 학습 금지" 조항 위반 위험.

### 3H단계 — 하이퍼파라미터 (valid 홀드아웃, 사용자 확정 A안)
문제 원문이 판단하라고 한 "배치 크기·동결할 층"과 옵티마이저·학습률. 3단계 승자를 기준으로 같은 규칙으로 판정. scene CV는 비용(설정당 약 5.5시간) 때문에 하지 않음.
| ID | 변경 | 근거 |
|---|---|---|
| H1 | SGD, lr0 0.01, momentum 0.937 | Ultralytics의 SGD 기본값. auto 규칙이 고른 AdamW와 비교 |
| H2 | AdamW lr0 0.001 (×0.5) | 학습률 양방향 비교 [내부: STRATEGY P1 학습률] |
| H3 | AdamW lr0 0.004 (×2) | 위와 같음 |
| H4 | `freeze=10` (백본 고정) | 원문 "동결할 층". 데이터가 적을 때 과적합 감소 가능, 학습 시간 감소 |
| H5 | batch 8 | 원문 "배치 크기" |
- 값(×0.5, ×2, 10층, 8)은 [판단]. 이웃 값과 동점일 가능성이 큼 [내부: STRATEGY 원칙 8].

### 4단계 — 추론·후처리 (재학습 없음, 저비용)
| ID | 변경 | 근거 |
|---|---|---|
| P-NMS | NMS IoU 0.7 → 0.5 / 0.3 (사전에 정한 두 값) | GT 최대 IoU 0.21 |
| P-TTA | 원본 + 상하 반전 예측을 WBF로 결합 | WBF는 TTA에서 AP50:95 +0.3 [문헌 6], 좌우 반전 제외 |
| P-ND | 예측 박스를 nodata가 아닌 영역의 경계로 자름 | nodata 경계에서 잘린 GT |

### 5단계 — 최종
- 고정된 설정으로 **fit+valid 재학습**(같은 E)한 모델을 **최종 제출**(사용자 확정). 이 모델의 valid 점수는 보고하지 않음.
- fit만으로 학습한 최종 설정 모델의 test 예측은 `submission_fitonly.csv`로 보존.
- `evaluation.validate(sub, test)`로 형식을 점검하고 `submission.csv`로 저장. 이전 제출은 `submission_v*.csv`로 보존.

## 5. 오류 분석 (판단 근거로 쓰지 않고 진단만)
- 크기 구간별(small/medium/large), 잘린 박스와 온전한 박스별, nodata 타일별 재현율 상한.
- AP50 − AP75 차이 → 위치 정밀도 문제인지 탐지 자체의 문제인지 구분.
- 배경 이미지의 오탐 수와 신뢰도 분포(`detections_on_negative_images`).
- 가장 나쁜 이미지 12장을 GT와 예측을 겹쳐 시각화.

## 6. 시간 예산 (추정, 0단계 측정 후 갱신)
- 학습 1회 ≤ 60분(B0 기준) × 후보 약 7개 + 최종 재학습 → 총 약 11~12시간, 여기에 3H단계(약 4~6시간)와 단계별 seed 재확인(단계당 약 2시간)을 더하면 **약 21~25시간** [추론]. 내역: B0·A1·C1·M11 각 약 55분, HN 약 83분(1.5배) + 채굴 추론 약 5분, R640 약 88분(1.6배), M8s 약 165~180분(3배), 최종 fit+valid 재학습 약 83분(이미지 약 1.5배). 같은 E로 학습하므로 비용 배수가 그대로 시간에 곱해짐. 측정값이 크게 다르면 3단계 M8s, 2단계 R640 순으로 줄임.
- **0단계 실측(2026-10-14)**: 1 epoch 7.6분(7스레드) → 규칙상 E = 10(최소값 적용), 학습 1회 약 76분 + 마지막 epoch 자체 평가 약 8분 + 예측. 위 내역의 55분을 약 85~95분으로 바꾸면 총 예상 시간은 **약 30~35시간** [추론]. → M8s 제외 + seed 재확인을 마지막에만 하기로 바꾼 뒤(2026-10-14) **약 13~17시간** [추론].
- 실제 시험(90분에 3문항)에 대비해 **빠른 경로**도 기록: B0 설정, E를 줄여 15분 안에 학습 → 제출. 이 경로의 valid 점수도 비교 기록으로 남김.

## 7. 일부러 하지 않는 것
- 잘린 박스나 작은 박스 삭제(평가 대상이며 라벨 규칙과 맞음).
- 회전 증강: 축 정렬 박스를 회전하면 박스가 부풀어 AP75 이상에서 손해 [추론].
- 좌우 반전 TTA, Ultralytics `augment=True`.
- image_id·scene_id 사용, test 재분할, test 의사 라벨.
- **타일 이어붙이기**(가장자리 매칭으로 주변 맥락 확보): 기술적으로는 가능하지만 scene/위치 정보를 쓰는 셈이라 규칙과 충돌할 위험이 있음 → 사용자 판단 전까지 제외.

## 8. 완료 기준
1. B0 제출 파일이 0단계에서 저장됨 → `validate` 통과.
2. 각 단계 판정표(짝 10개, 개선/악화/동점, 평균 d, pooled)가 `EXPERIMENTS.md`에 있음.
3. 최종 `submission.csv`가 `validate`를 통과하고, 학습·추론 스크립트와 로그(실행 시간 포함)가 문제 폴더에 남아 있음.

## 9. 레퍼런스
1. Purohit, Adler, Kerner. *ConeQuest: A Benchmark for Cone Segmentation on Mars*, WACV 2024 — https://arxiv.org/pdf/2311.08657 , supplement https://openaccess.thecvf.com/content/WACV2024/supplemental/Purohit_ConeQuest_A_Benchmark_WACV_2024_supplemental.pdf (CTX 5 m/px, 3개 지역 8개 하위 타일, 지역·크기 일반화가 어려움, Hypanis의 변동성이 큼, small cone이 가장 어려움)
2. Mars-Bench (NeurIPS 2025) — https://neurips.cc/virtual/2025/poster/121607 , Zenodo https://zenodo.org/records/15494675 , 리더보드 CSV https://huggingface.co/spaces/Mirali33/Mars-Bench/blob/main/data/detection.csv (conequest_det: RetinaNet 0.160, Faster R-CNN 0.132, SSD 0.013 mAP. 분할·지표 정의가 달라 직접 비교 불가)
3. Mars-Bench 소개 — https://arxiv.org/pdf/2510.24010
4. MOMO (CVPR 2026) — https://arxiv.org/abs/2604.02719 , supplement https://openaccess.thecvf.com/content/CVPR2026/supplemental/Purohit_MOMO_Mars_Orbital_CVPR_2026_supplemental.pdf (CTX 사전학습 = Murray Lab 전역 모자이크 → 누수 위험으로 제외)
5. Ultralytics YOLO11 vs YOLOv8 — https://docs.ultralytics.com/compare/yolo11-vs-yolov8 ; 학습 설정 — https://docs.ultralytics.com/usage/cfg
6. Solovyev et al. *Weighted Boxes Fusion* — https://arxiv.org/pdf/1910.13302
7. MSSS CTX 설명(오후 3시 궤도) — https://www.msss.com/mro/ctx/ctx_description.html
8. HiRISE 오후 촬영 예시(서쪽 조명) — https://hirise.lpl.arizona.edu/HiCat/images/PSP/PSP_001493_1815
9. CLAHE + YOLO(야간 교통, 위성 영상 아님) — https://www.ejournal.itn.ac.id/jati/article/view/18801
10. 타일 경계 처리(Flip-n-Slide) — https://arxiv.org/pdf/2404.10927 ; YOLT — https://arxiv.org/pdf/1805.09512
11. 크레이터 탐지 YOLO 비교 — https://arxiv.org/pdf/2405.16953 , https://arxiv.org/html/2508.03920v1
12. 내부 기록: `../STRATEGY.md` 2.2~2.5절·4장·7장(P11 조각 추론·경계 처리), `../problem_11/EXPERIMENTS.md`
