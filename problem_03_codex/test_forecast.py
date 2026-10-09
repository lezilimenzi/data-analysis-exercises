"""Tests of information boundaries and forecast/submission contracts."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from forecast import (check_submission, features, forecast_at, metrics, predict,
                      select_candidate, training_data)
from starter import save_submission


class ForecastTests(unittest.TestCase):
    def setUp(self):
        index = pd.date_range('2014-01-01', periods=24 * 20, freq='h')
        t = np.arange(len(index), dtype=float)
        self.data = pd.DataFrame({'temperature_c': 0.01*t + np.sin(t/4),
                                  'dew_point_c': -5 + np.cos(t/6),
                                  'pressure_hpa': 1020 + np.sin(t/20)}, index=index)
        self.origin = index[24 * 12 - 1]

    def test_training_boundary(self):
        history = self.data.loc[:self.origin]
        x, y, origins = training_data(history)
        self.assertEqual(origins[-1], self.origin - pd.Timedelta(hours=72))
        self.assertEqual(len(y), len(origins) * 72)
        self.assertEqual(len(x), len(y))
        self.assertEqual(y[-1], history.temperature_c.iloc[-1])
        # One fewer observed hour must exclude the entire last 72-label block.
        _, _, shortened = training_data(history.iloc[:-1])
        self.assertNotIn(origins[-1], shortened)

    def test_features_ignore_future(self):
        changed = self.data.copy()
        changed.loc[changed.index > self.origin] = 999999
        for auxiliary in (False, True):
            for linear in (False, True):
                np.testing.assert_array_equal(features(self.data, self.origin, auxiliary, linear),
                                              features(changed, self.origin, auxiliary, linear))

    def test_forecasts_ignore_future(self):
        changed = self.data.copy()
        changed.loc[changed.index > self.origin] = -999999
        candidates = [dict(kind='last'), dict(kind='repeat'), dict(kind='hourly', days=3),
                      dict(kind='pattern', window=24, tau=72),
                      dict(kind='ridge', alpha=1, auxiliary=True, winter=True),
                      dict(kind='boost', auxiliary=True, winter=True)]
        for candidate in candidates:
            a = forecast_at(self.data, self.origin, candidate)
            b = forecast_at(changed, self.origin, candidate)
            self.assertEqual(a.shape, (72,))
            self.assertTrue(np.isfinite(a).all())
            np.testing.assert_array_equal(a, b)

    def test_repeat24(self):
        history = self.data.iloc[:24].copy()
        history['temperature_c'] = np.arange(24)
        forecast = predict(history, dict(kind='repeat'))
        np.testing.assert_array_equal(forecast, np.tile(np.arange(24), 3))
        np.testing.assert_array_equal(forecast[[0, 23, 24, 71]], [0, 23, 0, 23])

    def test_selection_requires_both_periods_and_two_wins(self):
        rows = []
        for name, main, extra in [('base', [3, 3, 3], [3]*5),
                                  ('pattern', [2, 2, 3], [4]*5),
                                  ('reg', [2, 2, 3], [2]*5)]:
            for phase, values in [('main', main), ('extra', extra)]:
                rows.extend(dict(candidate=name, phase=phase, cutoff=i, mae=v)
                            for i, v in enumerate(values))
        scores = pd.DataFrame(rows).astype({'mae': float})
        selected, _ = select_candidate('base', 'pattern', 'reg', scores)
        self.assertEqual(selected, 'reg')
        scores.loc[(scores.candidate == 'reg') & (scores.phase == 'main'), 'mae'] = [0, 3.1, 3.1]
        selected, _ = select_candidate('base', 'pattern', 'reg', scores)
        self.assertEqual(selected, 'base')

    def test_metrics_and_submission(self):
        self.assertEqual(metrics(np.zeros(72), np.ones(72))['mae'], 1)
        sample = pd.DataFrame({'forecast_id': [f'WX-{i:03d}' for i in range(1, 73)]})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'submission.csv'
            save_submission(sample, np.arange(72), path)
            result = check_submission(path, sample)
            np.testing.assert_array_equal(result.temperature_c, np.arange(72))


if __name__ == '__main__':
    unittest.main()
