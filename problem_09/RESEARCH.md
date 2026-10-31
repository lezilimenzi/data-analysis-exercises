# RESEARCH — 문제 9 (7×7 격자 셀 분류, mIoU)

작성일 2026-10-09. 문제 패키지(`data/`, `README.md`, `feature_helper.py`, `models/`)만 사용. `problem_09_instructor`는 열지 않음.

## 1. 문제 요약
- 항공영상 1장(224×224 RGB)을 7×7=49칸(칸당 32×32px)으로 나누고, 칸마다 최대 면적 클래스를 예측.
- 클래스: 0 기타 / 1 침수 건물·도로 / 2 비침수 건물·도로 / 3 일반 수면·수영장.
- 평가: 모든 test 셀을 모아 클래스별 IoU = TP/(TP+FP+FN), 4개 평균(mIoU).
- 제약: CPU 전용, 처음부터 정답까지 30분. 패키지: numpy, pandas, sklearn 1.7.2, scipy, Pillow, onnxruntime 1.23.2. torch·onnx 패키지 없음, GPU 없음.
- 제출: `submission.csv` (`image_id,cell_id,label`, 9,359행, sample_submission 순서). 노트북 `problem.ipynb`는 패키지에 없어서 새로 만들어야 함.

## 2. 데이터 구조 (확인된 사실)
| 항목 | 값 |
|---|---|
| train | 765장 (fit 573 / valid 192), 셀 37,485개 |
| test | 191장, 셀 9,359개 |
| scene | train 64개 (fit 48 / valid 16), test 16개. **세 집합 간 겹침 없음**. scene당 약 12장 |
| 결측·중복·거의 같은 이미지 | 없음 |
| CSV 인코딩 | BOM 포함, `utf-8-sig`로 읽어야 함 |
| cell_id | 행 우선 (row=id//7, col=id%7), `training_examples.jpg`로 확인 |

## 3. 라벨 분포 (확인된 사실)
| 클래스 | 전체 | fit | valid |
|---|---|---|---|
| 0 | 63.7% | 62.2% | 68.1% |
| 1 | 14.1% | 12.7% | 18.1% |
| 2 | 5.3% | 5.0% | 6.2% |
| 3 | 16.9% | 20.1% | 7.5% |

- 전부 0인 이미지 258장(34%). 전부 0으로 예측하면 valid mIoU 0.1704.
- 1과 2가 같이 있는 이미지는 13/765장. 건물 셀의 97.7%가 자기 scene의 다수 침수 상태와 같음.
- 이웃 셀 라벨 일치율: 가로 86.5%, 세로 88.2%. 셀의 81.2%가 이미지 최빈 라벨과 같음.
- 위치(행·열)에 따른 클래스 비율 차이는 약 ±1%p.

## 4. 셀 색상 통계 (확인된 사실)
| 클래스 | 평균 RGB | 셀 내부 std |
|---|---|---|
| 0 | (105,114,81) | 20 |
| 1 | (114,114,102) | 24–25 |
| 2 | (168,170,170) | 21–25 |
| 3 | (114,108,93) | 12–13 |

이미지 평균 밝기 분산: fit 31.4 / valid 21.3 / test 36.6.

## 5. 인코더와 비용 (확인된 사실)
- `models/image_encoder.onnx`: ImageNet MobileNetV2. 출력은 1280차원 전역 풀링 벡터. **입력은 1×3×224×224로 고정**(배치 불가, 다른 크기 불가).
- 이미지 1장 추론: 6.1ms (스레드 2개). 전체 이미지 956장은 약 6초.
- 셀별 임베딩(셀을 224로 확대해 1회씩, 46,844회): 프로세스 4개 × 스레드 2개로 **182.5초**.
- HGB를 1289차원(손특징 9 + 셀 임베딩 1280)에 그대로 학습하면 6-fold에 약 340초. 30분 제한에서는 차원 축소가 필요함.

## 6. 탐색용 확인 (Probe) — 선택에 참고한 점수
모두 **fit 48개 scene만** 사용(scene 단위 GroupKFold). valid와 test는 쓰지 않음. 아래 점수는 후보 구성에 참고했으므로 **성능 확증 값이 아님**.

### 6-1. 셀 특징 비교 (GroupKFold 6, fold별 mIoU, d = 후보 − 손특징 9개)
| 모델 | a 손특징 9 | b 셀 임베딩 | c a+b | b−a (개선/악화) | c−a (개선/악화) |
|---|---|---|---|---|---|
| LR C=0.01 | 0.4389 | 0.4360 | 0.4366 | −0.0029 (3/3) | −0.0023 (3/3) |
| LR C=0.1 | 0.4490 | 0.4126 | 0.4135 | −0.0365 (1/5) | −0.0356 (1/5) |
| LR C=1 | 0.4509 | 0.4003 | 0.4015 | −0.0506 (1/5) | −0.0494 (1/5) |
| HGB(150, lr 0.1) | 0.4367 | 0.4613 | **0.4708** | +0.0247 (5/1) | **+0.0341 (6/0)** |

- 선형 모델에서는 셀 임베딩이 도움이 되지 않았고, HGB에서는 도움이 되었음.
- 모델 간 비교 HGB-c vs LR C=1-a: fold별 차이 [+.077, −.018, +.050, −.060, +.058, +.012], 평균 +0.020, 개선 4 / 악화 2.

### 6-2. 기준 모델(LR C=1, 손특징 9)의 오류 구조 (fit OOF)
| 실제 \ 예측 비율 | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 1 | **0.537** | 0.234 | 0.016 | **0.212** |
| 2 | 0.253 | 0.085 | 0.631 | 0.031 |
| 3 | **0.350** | 0.052 | 0.027 | 0.570 |

재현율: [0.889, 0.234, 0.631, 0.570].

### 6-3. 이미지 맥락으로 침수 구분 (전역 임베딩, LR C=0.01, GroupKFold 4)
- 대상: 건물 셀이 있는 fit 이미지 259장(37개 scene, 침수 59.8%).
- 이미지 단위 정확도 0.927 / AUC 0.961. 같은 scene 평균 임베딩 사용 시 정확도 0.954 / AUC 0.966.

### 6-4. test와 train의 유사성 (라벨 미사용)
- 임베딩 중심 거리: test–fit 5.78, test–valid 6.19, fit–valid 4.01.
- test 이미지의 최근접 train 이미지가 fit일 비율 0.712 (fit 비율 0.749).
- test는 fit과 valid 어느 쪽에도 특별히 가깝지 않음.

## 7. 가장 예측이 안 되는 부분 (해석)
1. **클래스 1(침수 건물·도로)**: IoU 0.17–0.23, 재현율 0.23. 주로 **0(54%)과 3(21%)으로 잘못 예측**되고, 2로 가는 비율은 1.6%뿐.
   - 해석(추론): 침수 지붕·도로는 색이 기타나 흙탕물과 비슷해서, 셀 하나만 보고는 "건물인지"도 "침수 장면인지"도 알기 어려움.
   - 이 이미지가 침수 장면이라는 맥락이 있으면 1과 3의 사전확률이 올라감. 전역 임베딩은 그 맥락을 AUC 0.96으로 잡아냄(6-3).
2. **클래스 3(수면)**: 35%가 0으로 예측됨. 홍수 물과 일반 수면이 같은 클래스라서 맥락과 매끈함이 단서.
3. **클래스 2**는 밝은 회색 지붕이라 셀 특징만으로도 비교적 잘 맞음(재현율 0.63).
4. fit과 valid의 클래스 구성 차이가 크고(클래스 3: 20% vs 7.5%), test scene은 모두 처음 보는 scene임. **scene 단위 일반화가 과적합의 주된 위험**.

## 8. 레퍼런스와 시사점
| 레퍼런스 | 내용 | 이 문제에 주는 시사점 |
|---|---|---|
| [FloodNet (Rahnemoonfar et al., IEEE Access 2021)](https://arxiv.org/abs/2012.02951) | 허리케인 Harvey 이후 UAV 영상. 침수·비침수 건물/도로, 수면, 수영장 등 픽셀 라벨 | 클래스 정의가 이 문제와 같아서 FloodNet 계열 데이터로 보임(추론, 미확인). 논문도 "자연 수면과 홍수 물 구분, 침수 건물·도로 검출"을 핵심 난제로 꼽음 |
| [FloodNet 10-class 라벨 (Zenodo)](https://zenodo.org/records/7566797) | building-flooded, road-non-flooded, pool 등 10개 클래스 | 이 문제는 그중 일부를 4개로 합친 것으로 추정 |
| [FloodNet Challenge 포럼](https://competitions.codalab.org/forums/26986/5525/) | "이미지 대부분이 홍수 물이면 Flooded"라는 라벨 기준 | 침수 여부가 이미지 단위 판단이라는 EDA 결과와 일치 |
| [Khose et al., NeurIPS 2021 CCAI](https://arxiv.org/abs/2105.08655) | FloodNet 분류(침수/비침수)와 분할을 함께 다룸 | 이미지 단위 침수 분류가 따로 학습 가능한 하위 과제임 |
| [Local-Global aerial segmentation (arXiv 1607.05620)](https://arxiv.org/abs/1607.05620) | 국소 외형 + 넓은 맥락 두 경로 결합 | 셀 특징 + 이미지 맥락 특징 결합의 근거 |
| [LANet (arXiv 1911.08877)](https://arxiv.org/abs/1911.08877) | 패치 단위 맥락 임베딩 강화 | 이웃·이미지 맥락 주입의 근거 |
| [Auto-context (Tu & Bai)](https://pages.ucsd.edu/~ztu/publication/pami_autocontext.pdf) | 1단계 확률맵을 2단계 특징으로 사용 | 이웃 확률을 쓰는 2단계 모델의 근거. 30분 제한 때문에 이번 계획에서는 단순 평활화로 대체 |
| [Huang et al. 2014 TGRS, 분류 후처리](https://www2.umbc.edu/rssipl/people/aplaza/Papers/Journals/2014.TGRS.Postprocessing.pdf) | 공간 정보로 픽셀 분류 잡음 제거 | 이웃 확률 평활화 후보의 근거. 얇은 구조(도로)가 지워질 위험도 있음 |
| [CNN Features off-the-shelf (Razavian et al. 2014)](https://arxiv.org/abs/1403.6382) | 고정된 사전학습 특징 + 단순 분류기가 강한 기준선 | 학습 없는 ONNX 특징 + HGB 접근의 근거 |
| [Classification refinement (arXiv 1801.07674)](https://arxiv.org/abs/1801.07674) | 전역 빈도 사전확률은 정확도는 올리지만 mIoU는 낮춤 | 드문 클래스에 유리하게 결정 경계를 옮겨야 mIoU가 오름 |
| [클래스별 임계값을 IoU로 조정 (arXiv 2111.08712)](https://arxiv.org/abs/2111.08712) | 검증 IoU 기준 클래스별 임계값 탐색 | 클래스별 로그확률 오프셋 조정 후보의 근거 |
| [Koyejo et al., NeurIPS 2015](https://papers.nips.cc/paper/2015/hash/85f007f8c50dd25f5a45fca73cad64bd-Abstract.html) | 집합형 지표의 최적 분류기는 확률 + 임계값(plug-in) 형태 | 확률 모델 학습 후 임계값만 따로 조정하는 2단계 방식의 이론 근거 |
| [Spatial CV (towardsdatascience)](https://towardsdatascience.com/spatial-cross-validation-using-scikit-learn-74cb8ffe0ab9/), [BORG spatial CV](https://gcol33.r-universe.dev/BORG/doc/spatial-cv.Rmd) | 공간 자기상관이 있으면 무작위 CV 점수가 부풀려짐 | scene 단위 GroupKFold 필수 |
| [Wolpert stacking 소개](https://wolpert.readthedocs.io/en/latest/user_guide/intro.html), [Ting & Witten 1999](https://cs.brynmawr.edu/Courses/cs372/spring2007/ting99issues.pdf) | 2단계 입력은 out-of-fold 예측이어야 함 | 오프셋 조정은 해당 fold를 뺀 OOF로만 해야 함 |
| [sklearn HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html) | class_weight는 1.2부터 지원 | 설치된 1.7.2에서 사용 가능 |

레퍼런스 한계: FloodNet 원본과 이 데이터의 관계는 클래스 정의가 같다는 것 외에 확인하지 못함. FloodNet 챌린지 우승 해법은 검색으로 찾지 못함.

## 9. 미확인 사항
- `onnx` 패키지 설치 가능 여부(가능하면 MobileNetV2의 7×7 특징맵을 이미지당 1회로 얻을 수 있음). 대회 환경 허용 여부 모름.
- 셀 임베딩을 PCA로 줄였을 때 HGB 이득이 유지되는지(미실험).
- 이미지 맥락 특징과 셀 특징을 결합한 효과(미실험).
