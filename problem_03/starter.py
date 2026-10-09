"""Input/output helpers; no fitted model or forecast is included."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent

def load_data(root=ROOT):
    root=Path(root)
    train=pd.read_csv(root/'train.csv',parse_dates=['timestamp'])
    test=pd.read_csv(root/'test.csv',parse_dates=['timestamp'])
    if not train.timestamp.is_monotonic_increasing or not train.timestamp.is_unique:
        raise ValueError('Training timestamps must be sorted and unique.')
    if not (train.timestamp.diff().dropna()==pd.Timedelta(hours=1)).all():
        raise ValueError('Training timestamps must be hourly.')
    expected=pd.date_range(train.timestamp.iloc[-1]+pd.Timedelta(hours=1),periods=72,freq='h')
    if len(test)!=72 or not np.array_equal(test.timestamp.to_numpy(),expected.to_numpy()):
        raise ValueError('Test must contain the next 72 hourly timestamps.')
    return train,test

def save_submission(test,predictions,path='submission.csv'):
    values=np.asarray(predictions,dtype=float)
    if values.shape!=(len(test),) or not np.isfinite(values).all():
        raise ValueError('One finite numeric prediction is required per test row.')
    if not test.forecast_id.is_unique:
        raise ValueError('Duplicate forecast IDs.')
    pd.DataFrame({'forecast_id':test.forecast_id,'temperature_c':values}).to_csv(path,index=False,encoding='utf-8-sig')

if __name__=='__main__':
    train,test=load_data()
    print('Train:',train.shape,'Test:',test.shape)
    print('Forecast origin:',train.timestamp.iloc[-1])
