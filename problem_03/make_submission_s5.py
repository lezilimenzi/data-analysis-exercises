"""S5 (0.5 * M2-ridge + 0.5 * analog ensemble) on all of train -> submission_02_claude_S5.csv.
Post-hoc strategy study (exp_strategy.py); the candidate list was motivated by test error analysis,
so a score for this file is not an independent evaluation."""
import numpy as np

import exp_strategy as E
import forecast as F
from starter import load_data, save_submission

train, test = load_data()
tabs = E.daily_tables(train)
ridge = E.cand_ridge(tabs)
assert np.allclose(ridge, F.m2_ridge(train, **F.M2_BEST))  # same as submission_01
analog = E.cand_s3(tabs, train)
pred = 0.5 * ridge + 0.5 * analog
save_submission(test, pred, 'submission_02_claude_S5.csv')

out = test.assign(ridge=ridge, analog=analog, S5=pred)
daily = out.groupby(out.timestamp.dt.date)
print(daily[['ridge', 'analog', 'S5']].agg(['min', 'max']).round(1))
print('mean |S5 - ridge| = %.2f, max = %.2f' % (np.abs(pred - ridge).mean(), np.abs(pred - ridge).max()))
