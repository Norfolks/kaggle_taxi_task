import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics

# Load raw data
# auto_data = pd.read_csv("regression/autos.csv", encoding="ISO-8859-1")

# Clear it
# auto_data = auto_data.fillna(auto_data.median().astype(np.int64))
# auto_data.to_csv("autos_with_median.csv", index=False)


auto_data = pd.read_csv("autos_with_median.csv")
auto_data = auto_data.fillna(auto_data.mode().dropna())
auto_data["notRepairedDamage"] = auto_data["notRepairedDamage"].map({'nein': 0, 'ja': 1, None: 2})

#Clear brands
# match = auto_data.brand.value_counts() < 1000
auto_data["brand"] = auto_data["brand"].apply(lambda x: "unknown" if not x else x)

# auto_data["name"] = auto_data["name"].apply(lambda x: 1 if '!' in x else 0)
# auto_data["postalCode"] = auto_data["postalCode"].apply(lambda x: x//10000)
auto_data = auto_data[auto_data.price > 200]
auto_data = auto_data[auto_data.price < 150000]
auto_data = auto_data[auto_data.powerPS > 10]
auto_data = auto_data[auto_data.yearOfRegistration < 2016]
auto_data = auto_data[auto_data.yearOfRegistration > 1950]
# auto_data = auto_data[auto_data.kilometer > 5000]
auto_data.drop([
    "postalCode",
    "seller",
    "lastSeen",
    "nrOfPictures",
    "dateCreated",
    "dateCrawled",
    "name",
    "model",
    'offerType',
],
    axis=1, inplace=True)
auto_data = auto_data.dropna()

# cols = auto_data.columns.tolist()
# cols = cols[0:2] + cols[3:] + [cols[2]]
# auto_data = auto_data[cols]

# hot_columns = []
# for col in auto_data.columns:
#     if auto_data[col].dtype == 'O':
#         hot_columns.append(col)
# hot_columns.remove("brand")
# auto_data = pd.get_dummies(auto_data, columns=hot_columns)

labelEncoder = preprocessing.LabelEncoder()
for col in auto_data.columns:
    if auto_data[col].dtype == 'O':
        labelEncoder.fit(np.unique(auto_data[col].unique()))
        transform_map = {key: labelEncoder.transform([key])[0] for key in auto_data[col].unique()}
        auto_data[col] = auto_data[col].map(transform_map)

auto_data.to_csv("autos_prepared_lenc.csv", index=False)

auto_data = pd.read_csv("autos_prepared_lenc.csv")


test_data = auto_data.sample(frac=0.33, random_state=3)
print(len(test_data.values))
auto_data = auto_data.drop(test_data.index)
print(len(auto_data.values))

data_y = auto_data.pop("price")
data_x = auto_data
test_y = test_data.pop("price")
test_x = test_data

dtrain = xgb.DMatrix(data_x.values, data_y.values)
dtest = xgb.DMatrix(test_x.values, test_y.values)

param = {'max_depth': 15, 'eta': 0.2, 'silent': 1, 'objective': 'reg:linear', 'booster': 'gbtree', 'seed': 1}
param['nthread'] = 4
param['eval_metric'] = 'rmse'
param['lambda'] = 40
param['alpha'] = 0
param['subsample'] = 1
param['gamma'] = 3000

param['n_estimators'] = 500
param['min_child_weight'] = 1

evalist = [(dtrain, 'train'), (dtest, 'test')]

prediction_model = xgb.train(param, dtrain, num_boost_round=100, evals=evalist)

predicted_y = prediction_model.predict(dtest)

# print("Train data:")
# print(data_y[:10].values)
# print(regressor.predict(data_x[:10]))

#
print("Test data:")
print(test_y.values[:10])
print(predicted_y[:10])



print("R2: {}".format(metrics.r2_score(test_y.values, predicted_y)))
print("neg_mean_absolute_error: {}".format(metrics.mean_absolute_error(test_y.values, predicted_y)))
print("neg_mean_squared_error: {}".format(metrics.mean_squared_error(test_y.values, predicted_y)))
print("explained_variance_score: {}".format(metrics.explained_variance_score(test_y.values, predicted_y)))
# xgb.plot_importance(prediction_model)
# print(data_x.head())
# plt.show()
