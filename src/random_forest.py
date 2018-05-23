from sklearn.ensemble import RandomForestRegressor

class RandomForestModel:
    def __init__(self):
        self.regressor = RandomForestRegressor(n_estimators=1000, max_depth=10, random_state=0)

    def get_params(self):
        return self.regressor.get_params()

    def train(self, X, y):
        self.regressor.fit(X, y)

    def predict(self, X):
        return self.regressor.predict(X.values.reshape(1, -1))

