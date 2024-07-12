import  pandas as pd

def data_load():
    data = pd.read_csv('testset.csv', index_col='datetime_utc')
    data.index = pd.to_datetime(data.index)
    return data

loading_data()
