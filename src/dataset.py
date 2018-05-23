import pandas as pd


class Dataset:
    def __init__(self, file_path):
        self.dframe = pd.read_csv(file_path, delimiter=';')
        self.adjust_time_format()

    def adjust_time_format(self):
        self.dframe['date/heure'] = pd.to_datetime(self.dframe['date/heure'])
        self.dframe.set_index('date/heure', inplace=True)

    def split_features_labels(self):
        X = self.dframe.drop('TEMP+1', axis=1)
        y = self.dframe['TEMP+1']
        return X, y

    def split_train_test(self, prop_train=0.8):
        X, y = self.split_features_labels()
        lim = int(len(y)*prop_train)
        X_train, X_test = X.iloc[:lim], X.iloc[lim:]
        y_train, y_test = y.iloc[:lim], y.iloc[lim:]
        return X_train, X_test, y_train, y_test

    def drop_nan(self):
        self.dframe.dropna(inplace=True)

    def add_feature(self, name, data):
        self.dframe[name] = data

    def drop_features(self, name_list):
        self.dframe.drop(name_list, axis=1, inplace=True)

    def get_last_features(self):
        X_t = self.dframe.drop('TEMP+1', axis=1).iloc[-1]
        return X_t

    def preprocess(self):
        print('Data engineering...')
        self.drop_features(['NO','NO2','CO2','PM10', 'HUMI'])
        self.add_feature('hour_of_day', self.dframe.index.hour)
        self.add_feature('day_of_month', self.dframe.index.day)
        self.add_feature('TEMP+1',self.dframe['TEMP'].shift(-1))
        self.add_feature('TEMP-1',self.dframe['TEMP'].shift(1))
