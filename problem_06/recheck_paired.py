"""Paired fold differences for STRATEGY.md grades (added 2026-10-02).
Part 1 (re-evaluation, Run 2/4): same candidates and folds as problem.ipynb cell 4 (seeds 2026-2028), plus the
Run 4 SVC C variants. Asserts fold means match EXPERIMENTS.md.
Part 2 (exploration, RESEARCH.md section 5): the original scratch scripts are lost. Rebuilt from the recorded settings and
accepted only if the recorded pooled scores (3 decimals) are reproduced; the seed set (42-44 or 100-102) is not
recorded, so both are tried and the one that reproduces is reported.
Output: paired_recheck.csv
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, TunedThresholdClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC

train = pd.read_csv("data/train.csv", encoding="utf-8-sig")
X, y = train["message"], train["label"].values
macro_f1 = make_scorer(f1_score, average="macro")
char_vec = lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)
word_vec = lambda: TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
tuned = lambda base: TunedThresholdClassifierCV(base, scoring=macro_f1, thresholds=41,
                                                cv=StratifiedKFold(3, shuffle=True, random_state=7))
svc = lambda c=1: LinearSVC(C=c, class_weight="balanced")


def run(cands, seeds):
    fold = {k: [] for k in cands}; oof = {k: np.zeros((len(seeds), len(y)), int) for k in cands}
    for r, s in enumerate(seeds):
        for a, b in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            for k, m in cands.items():
                p = clone(m).fit(X.iloc[a], y[a]).predict(X.iloc[b])
                oof[k][r, b] = p; fold[k].append(f1_score(y[b], p, average="macro"))
    pooled = {k: np.mean([f1_score(y, o, average="macro") for o in oof[k]]) for k in cands}
    return {k: np.array(v) for k, v in fold.items()}, pooled


rows = []
def add(part, fold, a, b):
    d = fold[b] - fold[a]; sd = d.std(ddof=1)
    rows.append(dict(part=part, baseline=a, candidate=b, n=len(d), mean=d.mean(), sd=sd, mean_over_sd=d.mean() / sd,
                     better=int((d > 0).sum()), worse=int((d < 0).sum()), ties=int((d == 0).sum())))

# Part 1: re-evaluation folds (notebook cell 4) + Run 4
A = make_pipeline(char_vec(), svc())
c1 = {"A": A, "Cthr": tuned(clone(A)),
      "Ethr": tuned(make_pipeline(make_union(word_vec(), char_vec()), svc())),
      "Cthr_C0.3": tuned(make_pipeline(char_vec(), svc(0.3))), "Cthr_C3": tuned(make_pipeline(char_vec(), svc(3)))}
f1, _ = run(c1, [2026, 2027, 2028])
for k, rec in {"A": 0.9818, "Cthr": 0.9832, "Ethr": 0.9824, "Cthr_C0.3": 0.9822, "Cthr_C3": 0.9836}.items():
    assert abs(f1[k].mean() - rec) < 0.0001, (k, f1[k].mean(), rec)
print("part 1 reproduced", {k: round(v.mean(), 4) for k, v in f1.items()}, flush=True)
for a, b in [("A", "Cthr"), ("A", "Ethr"), ("Cthr", "Cthr_C0.3"), ("Cthr", "Cthr_C3")]:
    add("re-evaluation (seeds 2026-2028)", f1, a, b)

# Part 2: exploration baselines (RESEARCH.md section 5)
c2 = {"word_NB": make_pipeline(word_vec(), MultinomialNB()),
      "word_LR": make_pipeline(word_vec(), LogisticRegression(class_weight="balanced", max_iter=1000)),
      "char_LR": make_pipeline(char_vec(), LogisticRegression(class_weight="balanced", max_iter=1000)),
      "char_SVC": make_pipeline(char_vec(), svc())}
rec2 = {"word_NB": 0.956, "word_LR": 0.961, "char_LR": 0.980, "char_SVC": 0.981}
for seeds in ([42, 43, 44], [100, 101, 102]):
    f2, pooled = run(c2, seeds)
    ok = all(abs(round(pooled[k], 3) - v) < 1e-9 for k, v in rec2.items())
    print("part 2 seeds", seeds, {k: round(v, 4) for k, v in pooled.items()}, "reproduced" if ok else "NOT reproduced", flush=True)
    if ok:
        tag = f"exploration (seeds {seeds[0]}-{seeds[-1]})"
        add(tag, f2, "word_NB", "char_SVC"); add(tag, f2, "word_LR", "char_LR")
        break

out = pd.DataFrame(rows)
out.to_csv("paired_recheck.csv", index=False)
print(out.round(4).to_string(index=False))
