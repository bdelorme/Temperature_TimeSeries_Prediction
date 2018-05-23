import sys
sys.path.insert(0, "../src/")
from src.random_forest import RandomForestModel
from src.dataset import Dataset

raw_data_file = 'data/qualite-de-lair-mesuree-dans-la-station-franklin-d-roosevelt_date_sorted.csv'

data = Dataset(raw_data_file)
data.preprocess()
X_t = data.get_last_features()
data.drop_nan()
X_train, y_train = data.split_features_labels()

model = RandomForestModel()
model.train(X_train, y_train)
prediction = model.predict(X_t)

print(prediction)
with open('result.txt', 'w') as f:
    f.write('%f' % prediction)

