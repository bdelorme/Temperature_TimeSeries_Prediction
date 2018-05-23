import pandas as pd

def read_raw_data(file_path):
    dframe = pd.read_csv(file_path, delimiter=';')
    adjust_time_format(dframe)
    dframe['TEMP+1'] = dframe['TEMP'].shift(-1)
    return dframe

def adjust_time_format(dframe):
    dframe['date/heure'] = pd.to_datetime(dframe['date/heure'])
    dframe.set_index('date/heure', inplace=True)

def split_features_label(dframe):
    y = dframe['TEMP+1']
    X = dframe.drop('TEMP+1', axis=1)
    return X, y

def split_train_test(X, y, prop_train=0.8):
    lim = int(len(y)*prop_train)
    X_train, X_test = X.iloc[:lim], X.iloc[lim:]
    y_train, y_test = y.iloc[:lim], y.iloc[lim:]
    return X_train, X_test, y_train, y_test

def preprocess_RF(dframe):
    dframe_RF = dframe.drop(['NO','NO2','CO2','PM10', 'HUMI'], axis=1)
    dframe_RF['hour_of_day'] = dframe_RF.index.hour
    dframe_RF['day_of_month'] = dframe_RF.index.day
    dframe_RF['TEMP-1'] = dframe_RF['TEMP'].shift(1)
    X_last = dframe_RF.drop('TEMP+1', axis=1).iloc[-1]
    dframe_RF.dropna(inplace=True)
    X, y = split_features_label(dframe_RF)
    return X, y, X_last

