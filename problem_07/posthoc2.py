"""7번 사후 보완 실험 v3 (PLAN.md v3). test 점수는 선택에 쓰지 않는다. 결과는 사후 성적.
실행: python3 posthoc2.py > posthoc2.out
"""
import time, warnings
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
warnings.filterwarnings('ignore')
T0 = time.time()
R = 'data/'
tr = pd.read_csv(R+'train.csv', encoding='utf-8-sig'); te = pd.read_csv(R+'test.csv', encoding='utf-8-sig')
se = pd.read_csv(R+'sessions.csv', encoding='utf-8-sig'); lg = pd.read_csv(R+'link_log.csv', encoding='utf-8-sig')
ss = pd.read_csv(R+'sample_submission.csv', encoding='utf-8-sig')
for c in ['rsrp_dbm','rsrq_db','snr_db','cqi','rssi_dbm','nr_rsrp_dbm']:
    lg[c] = pd.to_numeric(lg[c], errors='coerce')
lg['ldl'] = np.log1p(lg.dl_kbps); lg['low'] = (lg.dl_kbps < 20000).astype(int); lg['g5'] = (lg.network_mode == '5G').astype(float)
assert (lg.groupby('sample_id').relative_second.max() == 0).all()

def feats(g):
    g = g.sort_values('relative_second')
    r, d, low, dl = g.relative_second.values, g.ldl.values, g.low.values, g.dl_kbps.values
    m5, m15, p5 = r >= -4, r >= -14, (r >= -9) & (r <= -5)
    assert m5.any() and p5.any()
    run = 0                                            # P: 공백(초 차이 > 1)이나 정상 값에서 멈춤
    for i in range(len(r) - 1, -1, -1):
        if not low[i] or (i < len(r) - 1 and r[i + 1] - r[i] != 1): break
        run += 1
    hi = r[low == 0]
    tr_ = np.nonzero(low[1:] != low[:-1])[0] + 1        # T: 인접 관측 사이 전환 (공백 건너뛴 전환 포함)
    f = lambda c, m: np.nanmean(g[c].values[m].astype(float)) if np.isfinite(g[c].values[m].astype(float)).any() else np.nan
    pre = (r >= -19) & (r <= -5)
    o = {
        'dl_raw45': dl.mean(),
        # F7
        'ldl_m5': d[m5].mean(), 'ldl_m15': d[m15].mean(), 'ldl_m45': d.mean(), 'low_f45': low.mean(),
        'ldl_min15': d[m15].min(), 'ldl_sd15': d[m15].std(), 'slope15': np.polyfit(r[m15], d[m15], 1)[0],
        # P
        'low_run_s': run, 't_since_high': -hi.max() if len(hi) else 45, 'low_f5': low[m5].mean(),
        # T
        'n_down45': int(((low[1:] == 1) & (low[:-1] == 0)).sum()), 'n_up45': int(((low[1:] == 0) & (low[:-1] == 1)).sum()),
        't_since_cross': -r[tr_[-1]] if len(tr_) else 45,
        # D
        'd5': d[m5].mean() - d[p5].mean(), 'last_minus_min5': d[-1] - d[m5].min(),
        # A
        'la_m5': np.log1p(dl[m5].mean()), 'la_m15': np.log1p(dl[m15].mean()),
        # (b) 구간 정의용
        'arith_m5': dl[m5].mean(),
        # 참고 실험 R
        'mode_switch45': int((g.network_mode.values[1:] != g.network_mode.values[:-1]).sum()),
    }
    for c in ['rsrp_dbm','rsrq_db','snr_db','cqi','nr_rsrp_dbm','rssi_dbm','g5']:
        o[f'{c}_d'] = f(c, m5) - f(c, pre)
    return pd.Series(o)

X = lg.groupby('sample_id').apply(feats)
tr = tr.merge(se, on='session_id').join(X, on='sample_id'); te = te.merge(se, on='session_id').join(X, on='sample_id')
for d_ in (tr, te):
    d_['stat'] = (d_.mobility == 'stationary').astype(float)
    d_['stat_x_d5'] = d_.stat * d_.d5; d_['stat_x_run'] = d_.stat * d_.low_run_s

F7 = ['ldl_m5','ldl_m15','ldl_m45','low_f45','ldl_min15','ldl_sd15','slope15']
P = ['low_run_s','t_since_high','low_f5']; T = ['n_down45','n_up45','t_since_cross']
D = ['d5','last_minus_min5']; A = ['la_m5','la_m15']; I = ['stat','stat_x_d5','stat_x_run']
RREF = [f'{c}_d' for c in ['rsrp_dbm','rsrq_db','snr_db','cqi','nr_rsrp_dbm','rssi_dbm','g5']] + ['mode_switch45']
SETS = [('P', F7+P), ('D', F7+D), ('A', F7+A), ('T', F7+T), ('P+D+I', F7+P+D+I), ('ALL', F7+P+T+D+A)]   # 비용 순서
assert tr[sum([c for _, c in SETS], [])].notna().all().all() and te[sum([c for _, c in SETS], [])].notna().all().all()
print('feature summary (train):'); print(tr[P+T+D+A].describe().T[['mean','min','max']].round(3).to_string())

y = tr.label.values; g = tr.session_id.values; st = tr.stat.values == 1
TGRID = np.round(np.arange(0.30, 0.701, 0.05), 2); TOL = 1e-9
mf1 = lambda yt, p: f1_score(yt, p, labels=[0, 1], average='macro', zero_division=0)
pick_t = lambda yt, p: max(TGRID, key=lambda t: (round(mf1(yt, (p > t).astype(int)), 4), -abs(t - 0.5)))
def model(cols): return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))

def loso_prob(idx, cols):
    p = np.zeros(len(idx)); gi = g[idx]
    for s in pd.unique(gi):
        h = gi == s
        p[h] = model(cols).fit(tr.iloc[idx[~h]][cols], y[idx[~h]]).predict_proba(tr.iloc[idx[h]][cols])[:, 1]
    return p

def loso_pred(idx, cols, mode):
    """idx 안에서 LOSO 예측. mode '05' = 0.5 고정, 'nest' = 학습 세션 내부 LOSO로 임계값 선택."""
    pred = np.zeros(len(idx), int); gi = g[idx]; ts = []
    for s in pd.unique(gi):
        h = gi == s; a = idx[~h]
        p = model(cols).fit(tr.iloc[a][cols], y[a]).predict_proba(tr.iloc[idx[h]][cols])[:, 1]
        t = 0.5 if mode == '05' else pick_t(y[a], loso_prob(a, cols))
        ts.append(t); pred[h] = p > t
    return pred, ts

def sess_scores(idx, pred):
    gi = g[idx]; return np.array([mf1(y[idx][gi == s], pred[gi == s]) for s in pd.unique(gi)])

def judge(d):
    up, dn = int((d > TOL).sum()), int((d < -TOL).sum())
    return {'n': len(d), 'up': up, 'down': dn, 'tie': len(d) - up - dn, 'mean_d': d.mean(), 'pass': bool(d.mean() > TOL and up > dn)}

def select(results):
    """results: [(name, d_array)] 비용 순서. §5: 통과 후보 중 평균 d 최대, 1e-9 이내 동점은 비용 순서."""
    ok = [(i, n, d.mean()) for i, (n, d) in enumerate(results) if judge(d)['pass']]
    if not ok: return 'C'
    best = max(m for _, _, m in ok)
    return min([(i, n) for i, n, m in ok if best - m <= TOL])[1]

CANDS = [(f'{sn} {mode}', cols, mode) for sn, cols in SETS for mode in ['05', 'nest']]

# ===== 1) 전체 LOSO =====
ALL = np.arange(len(y))
predC, tsC = loso_pred(ALL, F7, 'nest'); sC = sess_scores(ALL, predC)
print(f'\n[재현] C pooled={mf1(y, predC):.4f} (기대 0.8339), thresholds={sorted(set(tsC))}  ({time.time()-T0:.0f}s)')
rows, PRED, DS = [], {'C': predC}, []
def row(name, pred, ts):
    d = sess_scores(ALL, pred) - sC; j = judge(d)
    warn = 'EDGE' if any(t in (0.30, 0.70) for t in ts) else ''
    rows.append({'cand': name, 'pooled': mf1(y, pred), 'mov': mf1(y[~st], pred[~st]), 'stat': mf1(y[st], pred[st]),
                 **{k: j[k] for k in ['up','down','tie','mean_d','pass']}, 'thr': ','.join(map(str, sorted(set(ts)))), 'warn': warn})
    return d
row('C (기준)', predC, tsC)
for name, cols, mode in CANDS:
    p, ts = loso_pred(ALL, cols, mode); PRED[name] = p; DS.append((name, row(name, p, ts)))
print(f'main done ({time.time()-T0:.0f}s)')
pd.set_option('display.width', 250)
res = pd.DataFrame(rows).set_index('cand')
print('\n=== 1. LOSO 결과 (판정: 세션 짝 §5, pooled는 보고만) ===')
print(res.round(4).to_string())
chosen = select(DS)
print(f'\n>>> 채택 (§5): {chosen}')

# ===== 2) 정지 세션별 FN/FP/재현율 =====
print('\n=== 2. 정지 세션별 (C vs 채택 후보 vs 각 후보의 FN/FP) ===')
def conf(p, m):
    yy = y[m]; pp = p[m]; tp = ((pp == 1) & (yy == 1)).sum()
    return f'FN{((pp==0)&(yy==1)).sum()} FP{((pp==1)&(yy==0)).sum()} rec{tp/max(1,(yy==1).sum()):.2f}'
for s in ['L017', 'L018', 'L019']:
    m = g == s
    print(s, f'n={m.sum()} pos={y[m].sum()} |', ' | '.join(f'{k}: {conf(v, m)}' for k, v in PRED.items()))

# ===== 3) (b) 구간 오류 변화 =====
b = ((tr.arith_m5 < 20000) & (tr.dl_raw45 >= 20000)).values
print(f'\n=== 3. (b) 구간: n={b.sum()}, pos={y[b].sum()} (C 오류 {(predC[b]!=y[b]).sum()}) ===')
for k, v in PRED.items():
    if k == 'C': continue
    gain = ((v == 1) & (predC == 0) & (y == 1) & b).sum(); newfp = ((v == 1) & (predC == 0) & (y == 0) & b).sum()
    lost = ((v == 0) & (predC == 1) & (y == 1) & b).sum(); fixfp = ((v == 0) & (predC == 1) & (y == 0) & b).sum()
    print(f'{k:14s} 새로 맞힌 양성 {gain}, 새 오탐 {newfp}, 놓친 양성 {lost}, 고친 오탐 {fixfp}, 순변화 {gain+fixfp-newfp-lost:+d}')

# ===== 4) 민감도: 평가에서 L011 / L018 제외 =====
print('\n=== 4. 민감도 (평가 집합에서만 제외, 재학습 없음) ===')
sess = pd.unique(g)
for ex in ['L011', 'L018']:
    keep = sess != ex
    out = []
    for name, d in DS:
        j = judge(d[keep]); out.append(f"{name}: {j['up']}/{j['down']}/{j['tie']} d={j['mean_d']:+.4f}{' P' if j['pass'] else ''}")
    print(f'without {ex}:'); print('   ' + '\n   '.join(out))

# ===== 5) 선택 절차 전체를 감싼 바깥 LOSO =====
print('\n=== 5. 선택 절차 전체를 감싼 추정 ===')
pred_w = np.zeros(len(y), int); picks = {}
for s in sess:
    h = g == s; a = ALL[~h]
    pc, _ = loso_pred(a, F7, 'nest'); sc_c = sess_scores(a, pc)
    inner = [(name, sess_scores(a, loso_pred(a, cols, mode)[0]) - sc_c) for name, cols, mode in CANDS]
    pick = select(inner); picks[s] = pick
    cols, mode = (F7, 'nest') if pick == 'C' else next((c, m) for n, c, m in CANDS if n == pick)
    p = model(cols).fit(tr.iloc[a][cols], y[a]).predict_proba(tr.iloc[ALL[h]][cols])[:, 1]
    t = 0.5 if mode == '05' else pick_t(y[a], loso_prob(a, cols))
    pred_w[h] = p > t
    print(f'  held-out {s}: picked {pick} (t={t})  ({time.time()-T0:.0f}s)', flush=True)
jw = judge(sess_scores(ALL, pred_w) - sC)
print(f'wrapped: pooled={mf1(y, pred_w):.4f} (C {mf1(y, predC):.4f}), vs C {jw}')

# ===== 6) 참고 실험 R (판정 제외) =====
print('\n=== 6. 참고 실험 R: F7 + 무선 변화량 + 모드 전환 (판정 제외) ===')
for mode in ['05', 'nest']:
    p, ts = loso_pred(ALL, F7 + RREF, mode); j = judge(sess_scores(ALL, p) - sC)
    print(f'R {mode}: pooled={mf1(y, p):.4f} stat={mf1(y[st], p[st]):.4f} vs C {j} thr={sorted(set(ts))}')

# ===== 7) 채택 후보가 C와 다르면 submission_v3.csv =====
print('\n=== 7. 제출 파일 ===')
if chosen == 'C':
    print('채택 후보 없음 → submission_v3.csv 생성하지 않음 (C 유지)')
else:
    cols, mode = next((c, m) for n, c, m in CANDS if n == chosen)
    t = 0.5 if mode == '05' else pick_t(y, loso_prob(ALL, cols))
    pt = (model(cols).fit(tr[cols], y).predict_proba(te[cols])[:, 1] > t).astype(int)
    sub = pd.DataFrame({'sample_id': ss.sample_id, 'label': pd.Series(pt, index=te.sample_id).loc[ss.sample_id].values})
    assert len(sub) == 168 and sub.label.isin([0, 1]).all()
    sub.to_csv('submission_v3.csv', index=False)
    old = pd.read_csv('submission.csv'); diff = old.label.values != sub.label.values
    print(f'submission_v3.csv: {chosen}, t={t}, positive rate {sub.label.mean():.3f}, C와 다른 예측 {diff.sum()}개')
    print(pd.DataFrame({'sample_id': sub.sample_id[diff], 'C': old.label[diff], 'v3': sub.label[diff]}).to_string(index=False))
print(f'\ntotal {time.time()-T0:.0f}s')
