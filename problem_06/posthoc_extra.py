"""사후 실험 (2026-10-05, test 채점 후. 제출 변경 없음). 판정 규칙: maicon/CLAUDE.md §5.
N1 NB-SVM, N2 문자 n-gram NB, K1 class_weight 없음 (seed 2026~2028, 15 fold, Cthr 대비)
D1 Cthr 보정 확률 + DistilBERT 확률 평균 (seed 2026, 5 fold, posthoc_bert 저장 확률 사용)
R1 연구 단계 기준선 재현 (seed 42~44, RESEARCH.md §5) → 짝지은 차이
출력: posthoc_extra.csv
"""
import warnings; warnings.filterwarnings("ignore")
import json, numpy as np, pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, TunedThresholdClassifierCV, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC
from joblib import Parallel, delayed

tr = pd.read_csv("data/train.csv", encoding="utf-8-sig")
X, y = tr["message"].values, tr["label"].values
mf = make_scorer(f1_score, average="macro")
char = lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)
tuned = lambda b: TunedThresholdClassifierCV(b, scoring=mf, thresholds=41, cv=StratifiedKFold(3, shuffle=True, random_state=7))

class NBScale(BaseEstimator, TransformerMixin):
    """NB 로그 비율 r = log((p/|p|1) / (q/|q|1))을 특징에 곱한다 (NB-SVM)."""
    def __init__(self, alpha=1.0): self.alpha = alpha
    def fit(self, X, y):
        p = self.alpha + np.asarray(X[y == 1].sum(0)).ravel()
        q = self.alpha + np.asarray(X[y == 0].sum(0)).ravel()
        self.r_ = np.log((p / p.sum()) / (q / q.sum())); return self
    def transform(self, X): return X @ sparse.diags(self.r_)

cands = {
    "Cthr": tuned(make_pipeline(char(), LinearSVC(C=1, class_weight="balanced"))),
    "N1 NB-SVM": tuned(make_pipeline(char(), NBScale(), LinearSVC(C=1, class_weight="balanced"))),
    "N2 char NB": tuned(make_pipeline(CountVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2), MultinomialNB())),
    "K1 no class_weight": tuned(make_pipeline(char(), LinearSVC(C=1))),
}
rows = []
def paired(part, base, cand, sb, sc):
    d = np.asarray(sc) - np.asarray(sb); tie = np.abs(d) <= 1e-9
    win, lose = int(((d > 0) & ~tie).sum()), int(((d < 0) & ~tie).sum())
    passed = d.mean() > 0 and win > lose
    rows.append(dict(part=part, baseline=base, candidate=cand, n=len(d), mean_base=np.mean(sb), mean_cand=np.mean(sc),
                     mean_d=d.mean(), better=win, worse=lose, ties=int(tie.sum()), rule_pass=passed))
    print(f"[{part}] {cand} vs {base}: {np.mean(sc):.4f} vs {np.mean(sb):.4f} | d {d.mean():+.4f} | 개선 {win} / 악화 {lose} / 동점 {int(tie.sum())} | 규칙 {'통과' if passed else '미통과'}", flush=True)

# --- N1, N2, K1 ---
splits = [(a, b) for r in range(3) for a, b in StratifiedKFold(5, shuffle=True, random_state=2026 + r).split(X, y)]
def run(k, a, b): return f1_score(y[b], clone(cands[k]).fit(X[a], y[a]).predict(X[b]), average="macro")
S = {k: Parallel(n_jobs=8)(delayed(run)(k, a, b) for a, b in splits) for k in cands}
assert abs(np.mean(S["Cthr"]) - 0.9832) < 5e-5, np.mean(S["Cthr"])  # EXPERIMENTS.md Run 2 재현
for k in ["N1 NB-SVM", "N2 char NB", "K1 no class_weight"]:
    paired("15fold seed2026-2028", "Cthr", k, S["Cthr"], S[k])

# --- D1 ---
sc, sd, sdb = [], [], []
for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=2026).split(X, y)):
    r = json.load(open(f"posthoc_bert/distilbert-base-uncased/fold{k}.json")); assert r["val_idx"] == b.tolist()
    pb = np.array(r["prob"])
    sc.append(f1_score(y[b], clone(cands["Cthr"]).fit(X[a], y[a]).predict(X[b]), average="macro"))
    cal = CalibratedClassifierCV(make_pipeline(char(), LinearSVC(C=1, class_weight="balanced")), method="sigmoid", cv=3).fit(X[a], y[a])
    pe = (cal.predict_proba(X[b])[:, 1] + pb) / 2
    sd.append(f1_score(y[b], (pe >= 0.5).astype(int), average="macro"))
    sdb.append(r["macro_f1"])
assert abs(np.mean(sc) - 0.9824) < 5e-5, np.mean(sc)  # posthoc_bert/cthr_folds.json 재현
paired("5fold seed2026", "Cthr", "D1 Cthr-cal + DistilBERT avg", sc, sd)

# --- R1 (RESEARCH.md §5 원래 코드) ---
W = lambda: TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
C = lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)
base = {
    "nb_word": make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), MultinomialNB(alpha=0.1)),
    "lr_word": make_pipeline(W(), LogisticRegression(C=10, max_iter=3000, class_weight="balanced")),
    "lr_char": make_pipeline(C(), LogisticRegression(C=10, max_iter=3000, class_weight="balanced")),
    "svc_char": make_pipeline(C(), LinearSVC(C=1, class_weight="balanced")),
}
recorded = {"nb_word": 0.9555, "lr_word": 0.9613, "lr_char": 0.9799, "svc_char": 0.9812}
F, ok = {}, True
for name, p in base.items():
    pooled, folds = [], []
    for rep in range(3):
        cv = StratifiedKFold(5, shuffle=True, random_state=42 + rep)
        pr = cross_val_predict(p, X, y, cv=cv); pooled.append(f1_score(y, pr, average="macro"))
        folds += [f1_score(y[b], pr[b], average="macro") for _, b in cv.split(X, y)]
    F[name] = folds; rep_ok = round(np.mean(pooled), 4) == recorded[name]; ok &= rep_ok
    print(f"[R1 재현] {name}: pooled {np.mean(pooled):.4f} (기록 {recorded[name]}) {'일치' if rep_ok else '불일치'}", flush=True)
if ok:
    paired("research seed42-44", "nb_word", "svc_char", F["nb_word"], F["svc_char"])
    paired("research seed42-44", "lr_word", "lr_char", F["lr_word"], F["lr_char"])
pd.DataFrame(rows).to_csv("posthoc_extra.csv", index=False)
