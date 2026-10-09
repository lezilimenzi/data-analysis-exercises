"""사후 실험 Run 11 (2026-10-05, test 채점 후. 제출 변경 없음). 판정: maicon/CLAUDE.md §5.
사전 고정 (결과 보기 전):
- 단서 묶음 6개 (아래 CLUES). 일반적인 SMS 스팸 단서로 정했지만, 놓친 스팸을 이미 읽은 뒤라 선택 편향이 있다.
- 후보 G1/G2: Cthr 문자 TF-IDF + 단서 지시변수 6개(가중치 1.0 / 0.3) → LinearSVC(C=1, balanced) + 중첩 임계값.
- 짝: seed 2026~2028 × 5-fold = 15. 기준 Cthr. 둘 다 통과하면 평균 d 최대.
- 진단: seed 2026 OOF에서 Cthr가 놓친 스팸을 단서 유무로 나눠 회복 수 비교. Ethr 대 Cthr 직접 짝 비교도 함께 계산.
출력: posthoc_clue.csv
"""
import warnings; warnings.filterwarnings("ignore")
import re, numpy as np, pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, TunedThresholdClassifierCV
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC
from joblib import Parallel, delayed

CLUES = {
    "billing": r"£|\b\d+p\b|\bp/|/msg|/min|per (min|msg|wk|week)|\bcost|\bcharge|\bchg\b|\bgbp\b|\bntwk\b|\bstd\b",
    "response": r"\b(txt|text|reply|send|sms)\b",
    "subscription": r"\b(stop|unsub\w*|subscri\w*|opt.?out|optout|end)\b",
    "url": r"www\.|https?://|\.co\.uk|\.com\b|\.net\b",
    "promo": r"\b(free|win|won|winner|prize|claim|award\w*|offer|cash|bonus|guaranteed)\b",
    "shortcode": r"(?<!\d)\d{5}(?!\d)",
}
class Clues(BaseEstimator, TransformerMixin):
    def __init__(self, w=1.0): self.w = w
    def fit(self, X, y=None): return self
    def transform(self, X):
        M = np.array([[bool(re.search(p, s, re.I)) for p in CLUES.values()] for s in X], float) * self.w
        return sparse.csr_matrix(M)

tr = pd.read_csv("data/train.csv", encoding="utf-8-sig")
X, y = tr["message"].values, tr["label"].values
mf = make_scorer(f1_score, average="macro")
char = lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)
word = lambda: TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
svc = lambda: LinearSVC(C=1, class_weight="balanced")
tuned = lambda b: TunedThresholdClassifierCV(b, scoring=mf, thresholds=41, cv=StratifiedKFold(3, shuffle=True, random_state=7))
cands = {
    "Cthr": tuned(make_pipeline(char(), svc())),
    "Ethr": tuned(make_pipeline(make_union(word(), char()), svc())),
    "G1 clues w1.0": tuned(make_pipeline(make_union(char(), Clues(1.0)), svc())),
    "G2 clues w0.3": tuned(make_pipeline(make_union(char(), Clues(0.3)), svc())),
}
splits = [(a, b) for r in range(3) for a, b in StratifiedKFold(5, shuffle=True, random_state=2026 + r).split(X, y)]
def run(k, a, b): return b, clone(cands[k]).fit(X[a], y[a]).predict(X[b])
S, P = {}, {}
for k in cands:
    out = Parallel(n_jobs=8)(delayed(run)(k, a, b) for a, b in splits)
    S[k] = np.array([f1_score(y[b], p, average="macro") for b, p in out])
    P[k] = np.zeros(len(y), int)
    for b, p in out[:5]: P[k][b] = p  # seed 2026
assert abs(S["Cthr"].mean() - 0.9832) < 5e-5 and abs(S["Ethr"].mean() - 0.9824) < 5e-5  # Run 2 재현

rows = []
def paired(base, cand):
    d = S[cand] - S[base]; tie = np.abs(d) <= 1e-9
    w, l = int(((d > 0) & ~tie).sum()), int(((d < 0) & ~tie).sum())
    fn, fp = int(((y == 1) & (P[cand] == 0)).sum()), int(((y == 0) & (P[cand] == 1)).sum())
    rows.append(dict(baseline=base, candidate=cand, n=15, mean_base=S[base].mean(), mean_cand=S[cand].mean(),
                     mean_d=d.mean(), sd_d=d.std(ddof=1), better=w, worse=l, ties=int(tie.sum()),
                     rule_pass=bool(d.mean() > 0 and w > l), fn=fn, fp=fp))
    print(f"{cand} vs {base}: {S[cand].mean():.4f} vs {S[base].mean():.4f} | d {d.mean():+.4f} (sd {d.std(ddof=1):.4f}) | "
          f"개선 {w} / 악화 {l} / 동점 {int(tie.sum())} | {'통과' if rows[-1]['rule_pass'] else '미통과'} | FN {fn} FP {fp}", flush=True)
paired("Cthr", "Ethr"); paired("Cthr", "G1 clues w1.0"); paired("Cthr", "G2 clues w0.3")

missed = (y == 1) & (P["Cthr"] == 0)
has = np.array([any(re.search(p, s, re.I) for p in CLUES.values()) for s in X])
print(f"Cthr 놓친 스팸 {missed.sum()}건: 단서 있음 {(missed & has).sum()} / 없음 {(missed & ~has).sum()}")
for k in ["G1 clues w1.0", "G2 clues w0.3"]:
    rec = missed & (P[k] == 1)
    print(f"  {k}: 회복 — 단서 있음 {(rec & has).sum()}, 단서 없음 {(rec & ~has).sum()} | 새로 놓친 스팸 {((y == 1) & (P['Cthr'] == 1) & (P[k] == 0)).sum()} | "
          f"새 오탐 {((y == 0) & (P['Cthr'] == 0) & (P[k] == 1)).sum()}")
print("정상 메시지 중 단서 있음:", int((has & (y == 0)).sum()), "/", int((y == 0).sum()), "| 스팸 중:", int((has & (y == 1)).sum()), "/", int((y == 1).sum()))
pd.DataFrame(rows).to_csv("posthoc_clue.csv", index=False)
