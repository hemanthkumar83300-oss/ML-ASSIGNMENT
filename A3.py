import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression

data = pd.read_csv("Coherence_bert_cls_embeddings.csv")

data['embedding'] = data['embedding'].apply(lambda x: eval(x))

X = np.array(data['embedding'].tolist())

y = data['label']

if y.dtype == 'object':
    y = pd.factorize(y)[0]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = X_train[:500]
y_train = y_train[:500]

print("Select regressors (comma separated numbers):")
print("1.LinearRegression  2.SVR  3.DecisionTree  4.RandomForest  5.AdaBoost  6.MLP")

choices = list(map(int, input().split(',')))

model_map = {
    1: ("LinearRegression", LinearRegression()),
    2: ("SVR", SVR()),
    3: ("DecisionTree", DecisionTreeRegressor()),
    4: ("RandomForest", RandomForestRegressor()),
    5: ("AdaBoost", AdaBoostRegressor()),
    6: ("MLP", MLPRegressor(max_iter=300))
}

results = []

for c in choices:
    name, model = model_map[c]

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    r2 = r2_score(y_test, y_test_pred)

    results.append([name, train_mse, test_mse, train_mae, test_mae, r2])

results_df = pd.DataFrame(results, columns=[
    "Model", "Train MSE", "Test MSE", "Train MAE", "Test MAE", "R2 Score"
])

print(results_df)
