The main premise for writing the code is as follows.

# Karpathy Guidelines

Behavioral guidelines to reduce common LLM coding mistakes, derived from [Andrej Karpathy's observations](https://x.com/karpathy/status/2015883857489522876) on LLM coding pitfalls.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Model Selection Rule (대회용)

"Simplicity First"는 코드에 관한 규칙이다. 모델·설정 선택에는 아래 규칙을 따른다.

- 후보와 짝 단위(fold, 반복, 평가 구간)는 결과를 보기 전에 정하고, 같은 분할에서 짝지어 비교한다.
- 짝별 차이 d_i는 후보가 나을수록 양수가 되게 계산한다(F1·정확도: 후보 − 기준, MAE 등 오차: 기준 − 후보).
  점수는 반올림 전 float64로 계산한다(torch 등에서 나온 float32 값은 float64로 바꾼 뒤 지표를 계산한다). |d_i| ≤ 1e-9이면 동점으로 센다.
  1e-9는 사전에 정한 수치 동점 허용오차다. "성능이 비슷하다"거나 "작은 개선은 무시한다"는 실용적 문턱이 아니다. 근거: P1~P8 기록에서 동점은 모두 정확히 0이었고, 0이 아닌 실제 차이 중 가장 작은 값은 1.85e-6(P5)이었다(2026-10-05 점검). 모든 지표에 맞는 보편값은 아니므로, 새 지표에서는 0이 아닌 차이의 최솟값이 이 값보다 충분히 큰지 확인한다.
- 채택 후보 조건: 동점을 포함한 전체 짝의 평균 d가 양수이고, 개선 짝 수가 악화 짝 수보다 많다(동점을 뺀 짝의 과반). 둘 중 하나라도 아니거나 전부 동점이면 기준 모델을 유지한다.
- 최종 선택: 채택 후보는 채택한다. 예외는 후보의 비용(구현·학습 시간, 버그 위험, 남은 시간)이 기대 이득보다 클 때뿐이며, 그 경우 이유를 기록한다. 계산·배포 예산을 넘는 후보는 결과를 보기 전에 정한 조건으로 제외한다.
- 여러 후보가 통과하면:
  - 후보, 평가 짝, 집계 방식, 비용 우선순위, 후보 순서를 결과를 보기 전에 고정한다. 모든 후보는 같은 기준 모델 대비 같은 짝으로 평가한다.
  - 전체 짝의 평균 d가 가장 큰 후보를 고른다. 같은 짝이면 평균 d의 차이가 곧 두 후보 간 짝별 차이의 평균이므로, 이것도 짝지은 비교다.
  - 최고 평균 d와의 차이가 1e-9 이내인 후보들은 동점으로 보고 비용 우선순위로 고른다. 비용도 같으면 고정된 후보 순서를 따른다.
  - 따로 통과한 후보들의 조합은, 결과를 보기 전에 조합 자체를 후보로 등록해 같은 짝으로 평가한 경우에만 고를 수 있다.
  - 후보 수를 줄여도 선택 편향은 없어지지 않는다. 고른 후보의 점수와 후보 간 차이는 선택에 사용한 점수라고 적고, 독립적인 성능 확증으로 보고하지 않는다.
- 전체 짝 수, 개선/악화/동점 수, 평균 d를 함께 보고한다. 동점이 대부분이면 두 모델의 예측이 거의 같다는 뜻이므로 기대 이득이 작다고 적는다.
- 이 조건은 운영상 기준이며 통계적 유의성이나 일반화 성능의 증거가 아니다.
- 결과를 본 뒤 이 규칙을 바꾸지 않는다. 바꿨다면 사후 변경으로 기록한다. test 점수로 다시 고르지 않는다.
