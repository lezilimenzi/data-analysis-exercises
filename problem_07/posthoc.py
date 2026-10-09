"""7번 사후 분석 (test 채점 0.9012 이후). test 점수는 선택에 쓰지 않는다.

사전 고정 (결과 보기 전):
- 평가: 14세션 LOSO. 기준 = 제출 모델 C (LR + 중첩 임계값). 첫 단계에서 C의 0.8339 재현.
- 후보마다 임계값 0.5 고정과 중첩 임계값(내부 LOSO, 0.30~0.70, 동점은 0.5 쪽)을 둘 다 계산.
- 등급: STRATEGY 0.1절. 짝 = 세션별 macro-F1 차이(14개). 단일 클래스 세션도 포함(zero_division=0).
- "후보가 C보다 낫다" 판단: 중첩 점수로 pooled 전체·test 비중 보정 모두 C 이상 + 등급 소폭 이상.
- 실행: python3 posthoc.py > posthoc.out
"""
import time, warnings
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
warnings.filterwarnings('ignore')
T0 = time.time()
R = 'data/'
tr = pd.read_csv(R+'train.csv', encoding='utf-8-sig'); te = pd.read_csv(R+'test.csv', encoding='utf-8-sig')
se = pd.read_csv(R+'sessions.csv', encoding='utf-8-sig'); lg = pd.read_csv(R+'link_log.csv', encoding='utf-8-sig')
lg['ldl'] = np.log1p(lg.dl_kbps); lg['low'] = (lg.dl_kbps < 20000).astype(float)

def feats(g):
    g = g.sort_values('relative_second')
    r, d = g.relative_second.values, g.ldl.values
    m5, m15 = r >= -4, r >= -14
    return pd.Series({
        'dl_raw45': g.dl_kbps.mean(),
        'ldl_m5': d[m5].mean(), 'ldl_m15': d[m15].mean(), 'ldl_m45': d.mean(),
        'low_f45': g.low.mean(),
        'ldl_min15': d[m15].min(), 'ldl_sd15': d[m15].std(),
        'slope15': np.polyfit(r[m15], d[m15], 1)[0],
        # 추가 표현 후보 (EDA에서 단변량 신호가 확인된 것만)
        'ldl_last': d[-1], 'low_f15': g.low.values[m15].mean(),
        'lul_m15': np.log1p(g.ul_kbps.values[m15]).mean(),
    })
X = lg.groupby('sample_id').apply(feats)
tr = tr.merge(se, on='session_id').join(X, on='sample_id')
te = te.merge(se, on='session_id').join(X, on='sample_id')
# 세션 맥락 (transductive): 같은 세션 전 샘플(train+test 입력 로그)의 ldl_m45 중앙값. 라벨은 쓰지 않음.
allx = pd.concat([tr[['session_id', 'ldl_m45']], te[['session_id', 'ldl_m45']]])
smed = allx.groupby('session_id').ldl_m45.median()
for d in (tr, te):
    d['sess_med'] = d.session_id.map(smed); d['dev_sess'] = d.ldl_m15 - d.sess_med

F7 = ['ldl_m5','ldl_m15','ldl_m45','low_f45','ldl_min15','ldl_sd15','slope15']
y = tr.label.values; g = tr.session_id.values; st = (tr.mobility == 'stationary').values
pt = (te.mobility == 'stationary').mean(); w = np.where(st, pt/st.mean(), (1-pt)/(1-st.mean()))
LOGO = list(LeaveOneGroupOut().split(tr, y, g)); SESS = pd.unique(g)
TGRID = np.round(np.arange(0.30, 0.701, 0.05), 2)
mf1 = lambda yt, p, sw=None: f1_score(yt, p, labels=[0,1], average='macro', zero_division=0, sample_weight=sw)
pick_t = lambda yt, p: max(TGRID, key=lambda t: (round(mf1(yt, (p > t).astype(int)), 4), -abs(t - 0.5)))
sess_scores = lambda p: np.array([mf1(y[g == k], p[g == k]) for k in SESS])

def lr(C=1.0): return lambda: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
def rf(seed=0): return lambda: RandomForestClassifier(500, min_samples_leaf=5, random_state=seed, n_jobs=-1)
def et(seed=0): return lambda: ExtraTreesClassifier(500, min_samples_leaf=5, random_state=seed, n_jobs=-1)
def gb(seed=0): return lambda: GradientBoostingClassifier(random_state=seed)

def oof(M, cols, idx=None, splits=LOGO):
    d = tr if idx is None else tr.iloc[idx].reset_index(drop=True)
    yy = d.label.values; p = np.zeros(len(d))
    for a, b in splits:
        p[b] = M().fit(d.iloc[a][cols], yy[a]).predict_proba(d.iloc[b][cols])[:, 1]
    return p

def nested(M, cols):
    pred = np.zeros(len(y), int); prob = np.zeros(len(y)); ts = []
    for a, b in LOGO:
        inner = list(LeaveOneGroupOut().split(a, y[a], g[a]))
        t = pick_t(y[a], oof(M, cols, a, inner)); ts.append(t)
        prob[b] = M().fit(tr.iloc[a][cols], y[a]).predict_proba(tr.iloc[b][cols])[:, 1]
        pred[b] = prob[b] > t
    return pred, prob, ts

def grade(d):
    n_up, n_dn = (d > 1e-9).sum(), (d < -1e-9).sum(); m, sd = d.mean(), d.std(ddof=1)
    if n_up >= 0.9*len(d) and m >= 3*sd: return '큰 개선'
    if n_up >= 2/3*len(d) and m > sd: return '뚜렷한 개선'
    if n_dn >= 2/3*len(d) or (m < 0 and -m > sd): return '악화'
    if n_up > n_dn and m > 0: return '소폭 개선'
    return '개선 근거 부족' + (' (평균 악화 방향)' if m < -1e-4 else '')

rows = []; PRED = {}
def report(name, pred, ref_sess):
    s = sess_scores(pred); d = s - ref_sess
    rows.append({'cand': name, 'all': mf1(y, pred), 'wtd': mf1(y, pred, w), 'mov': mf1(y[~st], pred[~st]),
                 'stat': mf1(y[st], pred[st]), 'up/down': f'{(d>1e-9).sum()}/{(d<-1e-9).sum()}',
                 'mean_d': d.mean(), 'sd_d': d.std(ddof=1), 'grade_vs_C': grade(d)})
    PRED[name] = pred

# 0) 재현
predA = (tr.dl_raw45 < 20000).astype(int).values
predC, probC, tsC = nested(lr(), F7)
sC = sess_scores(predC)
print(f'[재현] C LOSO all={mf1(y, predC):.4f} (노트북 0.8339), thresholds={sorted(set(tsC))}')
report('A rule45', predA, sC); report('C LR(C=1)+nested [제출]', predC, sC)

# 1) 후보: (이름, 모델, 피처)
CANDS = [
    ('LR C=0.1', lr(0.1), F7), ('LR C=10', lr(10), F7),
    ('RF', rf(), F7), ('ET', et(), F7), ('GB', gb(), F7),
    ('LR +last,low15', lr(), F7 + ['ldl_last', 'low_f15']),
    ('LR +ul', lr(), F7 + ['lul_m15']),
    ('LR +sess_ctx (transductive)', lr(), F7 + ['sess_med', 'dev_sess']),
    ('RF +sess_ctx (transductive)', rf(), F7 + ['sess_med', 'dev_sess']),
]
PROB = {'C': probC}
for name, M, cols in CANDS:
    p05 = (oof(M, cols) > 0.5).astype(int)
    pn, prob, ts = nested(M, cols); PROB[name] = prob
    report(f'{name} @0.5', p05, sC); report(f'{name} +nested', pn, sC)
    print(f'{name}: nested thresholds {sorted(set(ts))}  ({time.time()-T0:.0f}s)')

# 2) 상보적 앙상블: LR(C) + RF 확률 평균, 중첩 임계값
def nested_avg():
    pred = np.zeros(len(y), int)
    for a, b in LOGO:
        inner = list(LeaveOneGroupOut().split(a, y[a], g[a]))
        pin = (oof(lr(), F7, a, inner) + oof(rf(), F7, a, inner)) / 2
        t = pick_t(y[a], pin)
        pb = (lr()().fit(tr.iloc[a][F7], y[a]).predict_proba(tr.iloc[b][F7])[:, 1] +
              rf()().fit(tr.iloc[a][F7], y[a]).predict_proba(tr.iloc[b][F7])[:, 1]) / 2
        pred[b] = pb > t
    return pred
report('LR+RF avg +nested', nested_avg(), sC)

res = pd.DataFrame(rows).set_index('cand')
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 100)
print('\n=== LOSO 결과 (짝 = 세션별 macro-F1 차이, 기준 C) ===')
print(res.round(4).to_string())

# 3) 시드 민감도 (RF, 0.5 고정)
print('\n=== RF seed 민감도 (@0.5, pooled all) ===')
print({s: round(mf1(y, (oof(rf(s), F7) > 0.5).astype(int)), 4) for s in [0, 1, 2]})

# 4) 오류 공통성
errC = predC != y
print('\n=== 오류 공통성 ===')
print(f'C 오류 {errC.sum()}개 / {len(y)}')
allp = [v for k, v in PRED.items() if k != 'C LR(C=1)+nested [제출]']
common = np.all([p != y for p in allp], axis=0)
print(f'모든 후보가 공통으로 틀린 샘플 {common.sum()}개 (C 오류 중 {(common & errC).sum()}개 = {(common & errC).sum()/errC.sum():.0%})')
near = np.abs(np.log(tr.dl_raw45 / 20000)) < 0.5
print(f'C 오류 중 경계 근처(log거리<0.5) 비율 {near[errC].mean():.0%}, stationary 비율 {st[errC].mean():.0%}, FN {((predC==0)&(y==1)).sum()} / FP {((predC==1)&(y==0)).sum()}')

# 5) 학습 곡선 (LR, 0.5 고정): 바깥 fold의 학습 세션 중 일부만 사용
print('\n=== 학습 곡선 (LR@0.5, 학습 세션 비율별 pooled macro-F1, 5회 평균) ===')
rng = np.random.default_rng(0)
for frac in [0.5, 0.75, 1.0]:
    sc = []
    for rep in range(5 if frac < 1 else 1):
        pred = np.zeros(len(y), int)
        for a, b in LOGO:
            ss = pd.unique(g[a]); keep = rng.choice(ss, max(2, int(round(frac*len(ss)))), replace=False)
            aa = a[np.isin(g[a], keep)]
            if len(np.unique(y[aa])) < 2: aa = a
            pred[b] = lr()().fit(tr.iloc[aa][F7], y[aa]).predict_proba(tr.iloc[b][F7])[:, 1] > 0.5
        sc.append(mf1(y, pred))
    print(f'frac={frac}: {np.mean(sc):.4f} ± {np.std(sc):.4f}')
print(f'\ntotal {time.time()-T0:.0f}s')
