import datetime
from geopy.distance import vincenty

import xgboost as xgb
from xgboost import plot_importance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics

hot_columns = ['weekday']


def prune_data(data):
    data.drop('starting_street', axis=1, inplace=True)
    data.drop('end_street', axis=1, inplace=True)
    data.drop('street_list', axis=1, inplace=True)
    data.drop('distance_per_street_list', axis=1, inplace=True)
    data.drop('duration_per_street_list', axis=1, inplace=True)
    return data


def preproccess_data(data):
    pck_dt = data['pickup_datetime'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    data['month'] = pck_dt.apply(lambda x: x.month)
    data['day'] = pck_dt.apply(lambda x: x.day)
    data['hour'] = pck_dt.apply(lambda x: x.hour)
    data['minute'] = pck_dt.apply(lambda x: x.minute)
    data['weekday'] = pck_dt.apply(lambda x: x.weekday())
    raw_distance = []
    for plong, plat, dlong, dlat in zip(data['pickup_longitude'],
                                        data['pickup_latitude'],
                                        data['dropoff_longitude'],
                                        data['dropoff_latitude']):
        raw_distance.append(vincenty((plat, plong), (dlat, dlong)).miles)
    data['distance'] = raw_distance

    data.drop('store_and_fwd_flag', axis=1, inplace=True)
    data.drop('pickup_datetime', axis=1, inplace=True)
    return data


def score(preds, trues):
    return np.mean((np.log(np.array(preds) + 1) - np.log(np.array(trues) + 1)) ** 2) ** 0.5


# train_data = pd.read_csv('train_2.csv')
# train_data = prune_data(train_data)
# print("Start train data preprocces")
# train_data = preproccess_data(train_data)
# # train_data.drop('dropoff_datetime', axis=1, inplace=True)
# train_data.drop('id', axis=1, inplace=True)
# print("Finished")
# train_data.to_csv('pr_train.csv', index=False)
train_data = pd.read_csv('pr_train.csv')

# test_data = pd.read_csv('test_2.csv')
# test_data = prune_data(test_data)
# print("Start test data preprocces")
# test_data = preproccess_data(test_data)
# print("Finished")
# test_data.to_csv('pr_test.csv', index=False)
test_data = pd.read_csv('pr_test.csv')

test_ids = test_data.pop('id')

# tt_test = train_data.sample(frac=0.33, random_state=3)
# print(train_data.shape)
# train_data = train_data.drop(tt_test.index)
train_data = train_data[train_data.trip_duration > 30]
# print(train_data.shape)

print(train_data.dtypes)

train_y = np.log(train_data['trip_duration'] + 1)
train_x = train_data.drop("trip_duration", axis=1, inplace=False)
# tt_y = np.log(tt_test['trip_duration'] + 1)
# tt_x = tt_test.drop('trip_duration', axis=1, inplace=False)

evallist = [
    (train_x, train_y),
    # (tt_x, tt_y)
]
model = xgb.XGBRegressor(n_jobs=2, booster='gbtree', max_depth=20, n_estimators=120,
                         reg_lambda=1.5, gamma=10
                         )
model.fit(train_x, train_y, eval_set=evallist, eval_metric='rmse', verbose=True)

prediction = model.predict(test_data)
prediction = np.exp(prediction) - 1
pred_df = pd.DataFrame(prediction, index=test_ids, columns=['trip_duration'])
pred_df.to_csv('test_answers.csv')

plot_importance(model)
plt.show()
