import pandas as pd
import numpy as np
from data_preprocessing import data_preprocess
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_numeric_dtype

def data_preprocess():
    data = data_analysis()
    data.rename(columns={' _conds': 'conditions', ' _dewptm': 'dewpoint',
                     ' _fog': 'fog', ' _hail': 'hail', ' _heatindexm': 'heatindex', ' _hum': 'humidity',
                     ' _precipm': 'precipitation', ' _pressurem': 'pressure', ' _rain': 'rain', ' _snow': 'snow',
                     ' _tempm': 'temp', ' _thunder': 'thunder', ' _tornado': 'tornado', ' _vism': 'visibility',
                     ' _wdird': 'wdirdegrees', ' _wdire': 'winddirection', ' _wgustm': 'windgust',
                     ' _windchillm': 'windchill', ' _wspdm': 'windspeed'}, inplace=True)
    data.drop(columns=['precipitation', 'windchill', 'heatindex', 'windgust'], inplace=True)
    print(f'dataset shape (rows, columns) - {data.shape}')
    data = data.replace(to_replace = -9999, value = np.nan)
    data.ffill(inplace=True)
    print(data[data.isnull()].count())
    return data
def feature_engineering():

    data = data_preprocess()
    le=LabelEncoder()
    data['winddirection']=le.fit_transform(data['winddirection'])
    data['conditions']=le.fit_transform(data['conditions'])
    data = data.drop(columns=['fog', 'hail','rain','snow','thunder','tornado'],axis=1)
    def remove_outliers(data,par):
        z = np.abs(stats.zscore(data[par]))
        a=np.where(z > 3)
        for i in a[0]:
            if i in data.index:
                data=data.drop(index=i,inplace=True)
        return data 
    
    for j in data.columns:
        if is_numeric_dtype(data[j]): 
            data = remove_outliers(data,j)
    data = data.resample('D').mean().fillna(method='ffill')
    print(data.head())
    data.to_csv("cleaned_weather_series.csv",index=True)
    return data

feature_engineering()
