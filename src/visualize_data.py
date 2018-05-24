import pandas as pd
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.seasonal import seasonal_decompose
from collections import OrderedDict
import matplotlib.pyplot as plt
import datetime as dt

def time_series(dframe, year1, month1, day1, year2, month2, day2):
    date_lim_inf = dt.date(year1, month1, day1)
    date_lim_sup = dt.date(year2, month2, day2)
    f, axarr = plt.subplots(len(dframe.columns), sharex=True, figsize=(80, 60))
    for i in range(len(dframe.columns)):
        axarr[i].plot(dframe.index, dframe[dframe.columns[i]])
        axarr[i].set_title(dframe.columns[i], fontsize=60)
        axarr[i].xaxis.set_tick_params(labelsize=50)
        axarr[i].set_yticks([])
        axarr[i].set_xlim([date_lim_inf, date_lim_sup])
    return f

def global_infos(dframe):
    ord_dict = OrderedDict([('Type', dframe.dtypes.values),
        ('#NaN', dframe.isnull().sum().values),
        ('Min', dframe.min()),
        ('Max', dframe.max()),
        ('Mean', dframe.mean()),
        ('Median', dframe.median()),
        ('std', dframe.std())])
    return pd.DataFrame(data=ord_dict)

def correlation_features(dframe):
    return dframe.corr()

def autocorrelation_temp(dframe):
    return autocorrelation_plot(dframe['TEMP'].dropna())

def plot_prediction(y_train, y_test, y_pred_train, y_pred_test,
        year1, month1, day1, year2, month2, day2):
    date_lim_inf = dt.date(year1, month1, day1)
    date_lim_sup = dt.date(year2, month2, day2)
    plt.figure(figsize=(10, 3))
    plt.plot(y_train.index, y_train, label="train")
    plt.plot(y_test.index, y_test, label="test")
    plt.plot(y_train.index, y_pred_train, '--', label="prediction train")
    plt.plot(y_test.index, y_pred_test, '--', label="prediction test")
    plt.legend(loc=(1.01, 0))
    plt.xlabel("Date")
    plt.ylabel("TEMP")
    plt.xlim([date_lim_inf, date_lim_sup])

def decompose_plots(dframe):
    decomposition = seasonal_decompose(dframe.dropna()['TEMP'].values, freq=30*24)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    plt.subplot(411)
    plt.plot(dframe['TEMP'], label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal,label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()

