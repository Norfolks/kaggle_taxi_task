import xgboost as xgb
import pandas as pd

from sklearn import metrics

# Load data

cred_data = pd.read_csv("creditcard.csv")
test_data = pd.concat([cred_data[cred_data.Class == 1].sample(frac=0.33, random_state=3),
                       cred_data[cred_data.Class == 0].sample(frac=0.33, random_state=3)])
train_data = cred_data.drop(test_data.index)

train_y = train_data["Class"]
train_x = train_data.drop("Class", axis=1, inplace=False)
test_y = test_data["Class"]
test_x = test_data.drop("Class", axis=1, inplace=False)

evallist = [(train_x, train_y), (test_x, test_y)]

classifier = xgb.XGBClassifier(n_jobs=4)
classifier.fit(train_x, train_y, eval_set=evallist, eval_metric="auc", verbose=True)

predicted_y = classifier.predict(test_x)

print(metrics.confusion_matrix(test_y, predicted_y))
