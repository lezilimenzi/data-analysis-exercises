"""Final forecast: M2-ridge trained on all of train, origin 2014-12-28 23:00."""
import numpy as np
import pandas as pd

import forecast as F
from starter import load_data, save_submission

train, test = load_data()
pred = F.m2_ridge(train, **F.M2_BEST)
save_submission(test, pred, 'submission.csv')

sub = pd.read_csv('submission.csv', encoding='utf-8-sig')
assert list(sub.columns) == ['forecast_id', 'temperature_c'] and len(sub) == 72
assert sub.forecast_id.is_unique and set(sub.forecast_id) == set(test.forecast_id)
assert np.isfinite(sub.temperature_c).all()

dec = train[train.timestamp.dt.month == 12].temperature_c
out = test.assign(temperature_c=pred)
print('saved submission.csv:', sub.shape)
print('December observed range: %.0f ~ %.0f, forecast range: %.1f ~ %.1f'
      % (dec.min(), dec.max(), pred.min(), pred.max()))
daily = out.groupby(out.timestamp.dt.date).temperature_c
print(pd.DataFrame({'min': daily.min(), 'min_hour': out.loc[daily.idxmin(), 'timestamp'].dt.hour.values,
                    'max': daily.max(), 'max_hour': out.loc[daily.idxmax(), 'timestamp'].dt.hour.values}).round(1))
