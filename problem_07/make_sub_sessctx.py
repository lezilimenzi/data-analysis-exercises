"""사후 제출 후보: LR(C=1) + 세션 맥락 피처(sess_med, dev_sess) + 임계값(C와 같은 규칙).
기존 submission.csv(C, test 0.9012)는 덮어쓰지 않고 submission_sessctx.csv로 저장한다.
실행: python3 make_sub_sessctx.py
"""
src = open('posthoc.py', encoding='utf-8').read().split('# 0) 재현')[0]   # 데이터·피처·함수 정의만 재사용
exec(src)

COLS = F7 + ['sess_med', 'dev_sess']
M = lr(1.0)
t = pick_t(y, oof(M, COLS))                      # train 전체 LOSO OOF로 임계값 선택 (C와 같은 규칙)
model = M().fit(tr[COLS], y)
p_te = model.predict_proba(te[COLS])[:, 1]
pred = pd.Series((p_te > t).astype(int), index=te.sample_id)

ss = pd.read_csv(R + 'sample_submission.csv', encoding='utf-8-sig')
sub = pd.DataFrame({'sample_id': ss.sample_id, 'label': pred.loc[ss.sample_id].values})
assert len(sub) == 168 and sub.label.isin([0, 1]).all() and sub.sample_id.is_unique
sub.to_csv('submission_sessctx.csv', index=False)

old = pd.read_csv('submission.csv')
assert (old.sample_id.values == sub.sample_id.values).all()
diff = old.label.values != sub.label.values
print(f'threshold={t}, positive rate={sub.label.mean():.3f} (C: {old.label.mean():.3f})')
print(f'differs from submission.csv (C): {diff.sum()} / {len(sub)}')
print(sub.assign(mob=te.set_index('sample_id').loc[sub.sample_id, 'mobility'].values).groupby('mob').label.agg(['mean', 'size']))
print('changed rows:'); print(pd.DataFrame({'sample_id': sub.sample_id[diff], 'C': old.label[diff], 'sessctx': sub.label[diff]}).to_string(index=False))
