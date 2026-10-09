"""사후 실험 Run 12 (2026-10-05, test 채점 후. 제출 변경 없음). 판정: maicon/CLAUDE.md §5.
배경: Run 11 진단(seed 2026 OOF)에서 G1의 새 오탐은 주로 정밀도가 낮은 단서 묶음(응답 유도·경품)에서 나왔다.
사전 고정 (결과 보기 전):
- H1: G1과 같은 구조(문자 TF-IDF + 단서 지시변수, 가중치 1.0, LinearSVC + 중첩 임계값).
      단, 각 학습 fold 안에서 스팸 비율 ≥ 0.9인 단서 묶음만 쓴다(평가 fold 정답은 묶음 선택에 쓰지 않음).
- 정규식: Run 11의 과금 패턴 "per (min|msg|...)"에 단어 경계가 없어 "Super msg"가 잡혔다 → "\\bper"로 수정(결과를 보고 고친 버그 수정).
- 판정 분할: 지금까지 쓰지 않은 seed 3030~3032 × 5-fold = 15쌍, 기준 Cthr. G1은 참고용(같은 정규식 수정 적용).
출력: posthoc_clue2.csv
"""
import warnings; warnings.filterwarnings("ignore")
import re, numpy as np, pandas as pd
from scipy import sparse
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, TunedThresholdClassifierCV
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC
from joblib import Parallel, delayed

CLUES = {
    "billing": r"£|\b\d+p\b|\bp/|/msg|/min|\bper (min|msg|wk|week)|\bcost|\bcharge|\bchg\b|\bgbp\b|\bntwk\b|\bstd\b",
    "response": r"\b(txt|text|reply|send|sms)\b",
    "subscription": r"\b(stop|unsub\w*|subscri\w*|opt.?out|optout|end)\b",
    "url": r"www\.|https?://|\.co\.uk|\.com\b|\.net\b",
    "promo": r"\b(free|win|won|winner|prize|claim|award\w*|offer|cash|bonus|guaranteed)\b",
    "shortcode": r"(?<!\d)\d{5}(?!\d)",
}
def hits(X): return np.array([[bool(re.search(p, s, re.I)) for p in CLUES.values()] for s in X], float)
class Clues(BaseEstimator, TransformerMixin):
    def __init__(self, w=1.0, min_prec=None): self.w, self.min_prec = w, min_prec
    def fit(self, X, y=None):
        H = hits(X); self.keep_ = np.ones(H.shape[1], bool)
        if self.min_prec is not None:
            prec = (H[y == 1].sum(0)) / np.maximum(H.sum(0), 1); self.keep_ = prec >= self.min_prec
        return self
    def transform(self, X): return sparse.csr_matrix(hits(X)[:, self.keep_] * self.w)

tr = pd.read_csv("data/train.csv", encoding="utf-8-sig")
X, y = tr["message"].values, tr["label"].values
mf = make_scorer(f1_score, average="macro")
char = lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)
svc = lambda: LinearSVC(C=1, class_weight="balanced")
tuned = lambda b: TunedThresholdClassifierCV(b, scoring=mf, thresholds=41, cv=StratifiedKFold(3, shuffle=True, random_state=7))
cands = {
    "Cthr": tuned(make_pipeline(char(), svc())),
    "G1 all clues (ref)": tuned(make_pipeline(make_union(char(), Clues(1.0)), svc())),
    "H1 high-prec clues": tuned(make_pipeline(make_union(char(), Clues(1.0, 0.9)), svc())),
}
splits = [(a, b) for r in range(3) for a, b in StratifiedKFold(5, shuffle=True, random_state=3030 + r).split(X, y)]
def run(k, a, b):
    m = clone(cands[k]).fit(X[a], y[a]); kept = None
    if k.startswith("H1"):
        cl = m.estimator_.steps[0][1].transformer_list[1][1]; kept = tuple(np.array(list(CLUES))[cl.keep_])
    return b, m.predict(X[b]), kept
S, P, K = {}, {}, {}
for k in cands:
    out = Parallel(n_jobs=8)(delayed(run)(k, a, b) for a, b in splits)
    S[k] = np.array([f1_score(y[b], p, average="macro") for b, p, _ in out])
    P[k] = np.zeros(len(y), int)
    for b, p, _ in out[:5]: P[k][b] = p
    K[k] = Counter(kk for *_, kk in out)
rows = []
for cand in ["G1 all clues (ref)", "H1 high-prec clues"]:
    d = S[cand] - S["Cthr"]; tie = np.abs(d) <= 1e-9
    w, l = int(((d > 0) & ~tie).sum()), int(((d < 0) & ~tie).sum())
    fn, fp = int(((y == 1) & (P[cand] == 0)).sum()), int(((y == 0) & (P[cand] == 1)).sum())
    rows.append(dict(candidate=cand, mean_base=S["Cthr"].mean(), mean_cand=S[cand].mean(), mean_d=d.mean(), sd_d=d.std(ddof=1),
                     better=w, worse=l, ties=int(tie.sum()), rule_pass=bool(d.mean() > 0 and w > l), fn=fn, fp=fp))
    print(f"{cand} vs Cthr: {S[cand].mean():.4f} vs {S['Cthr'].mean():.4f} | d {d.mean():+.4f} (sd {d.std(ddof=1):.4f}) | "
          f"개선 {w} / 악화 {l} / 동점 {int(tie.sum())} | {'통과' if rows[-1]['rule_pass'] else '미통과'} | FN {fn} FP {fp} (seed 3030)")
print("Cthr seed 3030 OOF: FN", int(((y == 1) & (P["Cthr"] == 0)).sum()), "FP", int(((y == 0) & (P["Cthr"] == 1)).sum()))
print("H1이 fold마다 남긴 묶음:", dict(K["H1 high-prec clues"]))
pd.DataFrame(rows).to_csv("posthoc_clue2.csv", index=False)
