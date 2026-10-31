# RESEARCH — 문제 10 (영문 보고 5분야 다중 라벨 분류, Macro-F1)

작성일 2026-10-10. 문제 패키지(`data/`, `README.md`, `requirements.txt`)만 사용. `problem_10_instructor`는 열지 않음.

## 1. 문제 요약
- 입력: 영문 보고 문단 `text` (+ `message_id`, `document_id`).
- 출력: 5개 분야 각각 0/1 — food, health, shelter, wash, logistics. 한 행에 여러 개 1 가능(다중 라벨).
- 주석: 1 = 해당 주석 있음, 0 = 없음, **-1 = 미검토**(제3 클래스도 0도 아님. 학습·평가에서 해당 칸만 제외).
- 평가: 분야별 F1 = 2TP/(2TP+FP+FN), 5개 단순 평균(Macro-F1).
- 제약: **CPU 전용**, 처음부터 정답까지 30분. requirements: numpy, pandas, scikit-learn, notebook.
- 제출: `submission.csv` = `message_id,food,health,shelter,wash,logistics` (1500행, sample_submission과 같은 순서 — 확인됨).

## 2. 환경 (확인된 사실)
| 항목 | 값 |
|---|---|
| 인터프리터 | `python3`만 있음 (`python` 없음) |
| 라이브러리 | sklearn 1.7.2, pandas 2.3.2, torch 2.10 (CUDA 없음), transformers, sentence-transformers. lightgbm 없음 |
| HF 캐시 | all-MiniLM-L6-v2, all-mpnet-base-v2, bert-base-uncased, distilbert-base-uncased |
| CPU | nproc 8, torch 기본 스레드 4 |
| CSV | 모두 UTF-8 BOM → `encoding='utf-8-sig'` |

## 3. 분할 구조 (확인된 사실)
| | 행 | 문서 | -1 존재 |
|---|---|---|---|
| fit | 6000 | 2859 | 있음 (모든 -1이 fit에 있음) |
| valid | 1500 | 411 | **없음** (5분야 전부 검토) |
| test | 1500 | 367 | – |

- fit / valid / test 간 `document_id` 겹침 **0** → 문서 단위 분할.
- valid와 test는 크기(1500)와 문서당 메시지 수(평균 3.6 / 4.1, 중앙값 2 / 2)가 비슷함. **추론**: test도 valid와 같은 방식(문서 분리, 전부 검토)으로 만들어졌을 가능성이 높음. 확인 불가.
- 중복 텍스트: train 안 0, test 안 0, train∩test 0.
- message_id 번호 구간별 양성률·검토율은 고르게 분포함 → 번호 순서 신호 없음.

## 4. 라벨 (확인된 사실)
| 분야 | 양성 | 검토된 행 | 양성률 | -1 비율 | fit 양성 | valid 양성 |
|---|---|---|---|---|---|---|
| food | 561 | 5328 | 10.5% | 29% | 401 | 160 |
| health | 1325 | 5421 | 24.4% | 28% | 940 | 385 |
| shelter | 274 | 4542 | 6.0% | 39% | 181 | 93 |
| wash | 377 | 4502 | 8.4% | 40% | 251 | 126 |
| **logistics** | **84** | 3882 | **2.2%** | 48% | **48** | **36** |

- 5분야 모두 -1인 행 106개(학습 정보 없음). 5분야 모두 검토된 행 1745개(그중 1500개가 valid).
- 행당 양성 수: 0개 5147 / 1개 2130 / 2개 이상 223.
- 동시 양성: health–wash 84, food–health 69, 나머지는 약함 → 분야별 독립 이진 분류(binary relevance)로도 손실이 작을 것(추론).
- **-1의 성격**: 분야 간 검토 여부가 거의 독립(P(b 검토|a 검토) 0.56–0.78 ≈ 단순 검토율). 어떤 분야가 -1인 행에서 다른 분야 양성률이 오히려 약간 낮음(예: wash -1일 때 0.242, 검토 시 0.303) → -1이 "숨은 양성"이라는 신호 없음. **분야별 마스킹**이 맞음.

## 5. 문서 구조 (확인된 사실)
- train 문서당 메시지 중앙값 1, 단 **D00000 하나가 698행**(9.3%, 전부 fit). D00000 양성률: health 29%, logistics 5.2%로 전체와 다름.
- 2행 이상 문서에서 라벨이 문서 안에서 모두 0이거나 모두 1인 비율: food 0.74, health 0.59, shelter 0.82, wash 0.74, **logistics 0.95** → 문서 단위 상관 강함. 무작위 KFold는 낙관적 → **GroupKFold(document_id) 필수**.
- test 안에도 다행 문서가 있음(D00887 105행, D03012 67행 …) → 같은 문서 예측을 평균하는 평활화의 여지(효과 미검증).
- 문서 안 검토 여부(-1) 일관성은 약함(all-or-none 45–50%).

## 6. 텍스트 (확인된 사실)
- 평균 55단어(train) / 51단어(test), 최대 약 300단어. 비ASCII 거의 없음.
- MiniLM 토크나이저 기준 토큰 길이: 중앙값 68, p90 128, p99 208, 최대 407. 128 초과 9.9%, 256 초과 0.3%.
- `[날짜, 지역]` 접두어: train 34%, test 22%(분포 차이). 지역: Bangladesh/Cox's Bazar, Nigeria, Syria(NES/NWS/GoS), Libya. 연도는 2020–2021이 대부분.
- 키워드 규칙 F1(검토된 행 기준, 참고용): wash 0.77, health 0.62, food 0.55, **shelter 0.29**(재현율 0.85, 정밀도 0.17 — "camp/displaced" 등이 흔함), **logistics 0.19**(재현율 0.36).
- logistics 양성 예: 수에즈 운하 선박 운항, 의약품 트럭 운송비, UNHAS 항공·화물 수송, 루머/긴장 관련 문장 등 → **의미 범위가 넓고 애매함**.

## 7. 빠른 기준 점수 (fit으로 학습 → valid로 평가, -1 마스킹)
TF-IDF word(1,2) + LR(C=4, class_weight=balanced), 실행 2.2초.

| | food | health | shelter | wash | logistics | Macro |
|---|---|---|---|---|---|---|
| F1 @0.5 | 0.738 | 0.710 | 0.588 | 0.669 | **0.053** | **0.552** |
| valid에서 고른 최적 임계값(낙관적) | 0.754 | 0.729 | 0.610 | 0.722 | 0.385 | ≈0.64 |

- 약점 1순위는 **logistics**(양성 48개로 학습, 임계값 0.5에서 거의 붕괴). 2순위는 **shelter**.
- 임계값 효과가 logistics 하나에 집중됨. 단, 위 최적값은 valid로 고른 것이라 선택 편향이 있음(독립 확증 아님).

## 8. 시간 측정 (확인된 사실, 이 PC 기준)
| 작업 | 시간 |
|---|---|
| word TF-IDF 적합(train+test 9000문장) | 1.7초 |
| char_wb(2–5) TF-IDF 적합 | 9.1초 |
| word LR 5라벨 학습(7500행) | 1.0초 |
| word+char LR 5라벨 학습(7500행) | 3.3초 |
| **MiniLM-L6 임베딩 9000문장(4스레드, max_seq 256)** | **실측 318.6초**. 재측정 추정 452초 |
| MiniLM, max_seq 128, 4스레드 | 추정 348초 |
| MiniLM, 8스레드 | 추정 615–1263초(오히려 느림, WSL 스레드 경합으로 추정) |
| mpnet-base 9000문장 | 추정 약 2483초(41분) → 사용 불가 |

- MiniLM은 30분 중 **5–8분**을 쓰고, 측정할 때마다 시간이 크게 달라짐. requirements.txt 밖 의존성(torch, sentence-transformers)이라 실제 시험 환경에 없을 위험도 있음.

## 9. 레퍼런스와 이 문제에 주는 시사점
| 레퍼런스 | 내용 | 시사점 |
|---|---|---|
| Durand et al., CVPR 2019 ([pdf](https://openaccess.thecvf.com/content_CVPR_2019/papers/Durand_Learning_a_Deep_ConvNet_for_Multi-Label_Classification_With_Partial_Labels_CVPR_2019_paper.pdf)); Ben-Baruch et al., CVPR 2022 ([pdf](https://openaccess.thecvf.com/content/CVPR2022/papers/Ben-Baruch_Multi-Label_Classification_With_Partial_Annotations_Using_Class-Aware_Selective_Loss_CVPR_2022_paper.pdf)); Huynh & Elhamifar, CVPR 2020 ([pdf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Huynh_Interactive_Multi-Label_CNN_Learning_With_Partial_Labels_CVPR_2020_paper.pdf)) | 부분 주석 다중 라벨: 미주석을 음성으로 두면 노이즈로 성능이 떨어짐. 기본은 미주석 무시(마스킹) | -1은 분야별로 마스킹. 분야별 독립 이진 분류이므로 마스킹이 곧 "그 분야 학습에서 행 제외" |
| Lipton, Elkan, Narayanaswamy, ECML PKDD 2014 ([arXiv 1402.1892](https://arxiv.org/abs/1402.1892)) | 보정된 확률이면 F1 최적 임계값 = 최적 F1의 절반. 희귀 라벨일수록 F1이 임계값에 민감 | 0.5 고정은 F1에 부적합. logistics는 임계값 조정이 핵심 |
| scikit-learn TunedThresholdClassifierCV ([docs](https://scikit-learn.org/stable/modules/classification_threshold.html)) | CV로 결정 임계값을 고르는 표준 방식 | 임계값은 학습에 쓰지 않은 OOF 예측으로만 고름. -1 마스킹·그룹 때문에 직접 구현 |
| Forman & Scholz, SIGKDD Explorations 2010 ([pdf](https://www.dcc.fc.up.pt/~ines/aulas/1516/DM1/Papers/v12-1-p49-forman-sigkdd.pdf)) | F1을 fold별로 평균하면 불균형이 심할 때 편향됨. TP/FP/FN을 모아서 계산 권장 | 임계값 선택은 여러 fold OOF를 **모은** 표본에서 함. fold별 점수는 짝 비교용으로만 씀 |
| Cawley & Talbot, JMLR 2010 ([link](https://www.jmlr.org/papers/v11/cawley10a.html)) | 모델 선택 기준의 분산이 크면 선택 과정 자체가 과적합되고, 고른 모델의 점수가 낙관적으로 편향됨 | 후보 수를 적게 고정, 하이퍼파라미터 탐색 안 함, 고른 점수는 "선택에 쓴 점수"로 보고 |
| Wang & Manning, ACL 2012 ([P12-2018](https://aclanthology.org/P12-2018.pdf)) | 짧은 텍스트에서 bigram, NB 가중(NBSVM)이 강한 기준선 | word(1,2) TF-IDF LR을 기준으로 하고 NB-LR을 후보로 둠 |
| 희소+조밀 특징 조합 ([arXiv 1708.03940](https://arxiv.org/pdf/1708.03940)); TF-IDF 대 임베딩 비교 ([arXiv 2412.21022](https://arxiv.org/html/2412.21022v1)) | 데이터·과제에 따라 우열이 갈림. 소량 데이터에서는 TF-IDF LR이 이기는 경우가 많음 | 임베딩 이득은 불확실. 비용(5–8분)이 확실하므로 예산 규칙으로 판단 |
| SetFit, Tunstall et al., NeurIPS 2022 ([blog](https://huggingface.co/blog/setfit)) | 소량 라벨에서 sentence-transformer 대조학습 미세조정이 강함 | GPU 없음 + MiniLM 순전파만으로 5분 이상 → 30분 안에서는 불가 |
| 희귀 라벨 임계값 실무 정리 ([MLM](https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/)); 불균형 텍스트 메타학습 ([arXiv 2310.15019](https://arxiv.org/pdf/2310.15019)) | 라벨별 임계값(CS-cut)이 공통 임계값보다 과적합되기 쉬움 | logistics 임계값은 **거친 격자(0.05 간격)** + 여러 fold를 모은 OOF로만 고름 |
| Read et al., Classifier Chains, MLJ 2011 ([link](https://mlanthology.org/mlj/2011/read2011mlj-classifier)) | 라벨 상관이 강할 때 체인이 도움. 앞 단계 오류가 전파됨 | 분야 간 상관이 약해(§4) 체인은 후보에서 제외 |
| Saerens et al. EM / Alexandari et al. 2020 ([arXiv 1901.06852](https://arxiv.org/abs/1901.06852)) | 사전확률이 바뀌면 보정된 확률로 조정 | fit·valid 양성률이 거의 같아(§4) 사전확률 보정은 불필요로 판단 |
| HumSet, Fekih et al., EMNLP Findings 2022 ([arXiv 2210.04573](https://arxiv.org/abs/2210.04573)) | DEEP 플랫폼의 인도지원 문서 스니펫 다중 라벨 데이터셋 | **추론**: 이 데이터가 DEEP/HumSet 계열일 가능성(문서-스니펫 구조, `[날짜, 지역]` 접두어). 라벨 체계 일치 여부는 확인하지 못함. 외부 데이터 사용 규정이 불명확해 사용하지 않음 |

## 10. 정리
- **확인된 사실**: -1은 fit에만 있음 / valid는 전부 검토됨 / 문서 단위 분할 / logistics 양성 84개 / CPU 전용 / MiniLM 임베딩 5–8분 / TF-IDF LR 학습은 수 초.
- **추론**: test는 valid와 같은 생성 방식일 것 / 승부처는 logistics의 임계값과 학습 양성 수(valid 합치면 48→84) / 문서 평활화가 logistics·shelter에 도움이 될 수 있음.
- **추가 검증 필요**: 문서 평활화·문자 n-gram·NB 가중의 실제 효과(PLAN의 사전 등록 비교로 확인). OOF(학습 80%) 기준 임계값이 전체 학습 모델에도 맞는지.
