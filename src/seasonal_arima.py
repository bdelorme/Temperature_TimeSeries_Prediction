import statsmodels.api as sm

class SARIMA:
    def __init__(self, data):
        self.regressor = sm.tsa.statespace.SARIMAX(data, order=(1,1,1), seasonal_order=(0,1,1,24))

    def train(self):
        self.regressor = self.regressor.fit(disp=0)

    def predict(self):
        return self.regressor.forecast(step=7)[0]

