import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("Coherence_bert_cls_embeddings.csv")

data['embedding'] = data['embedding'].apply(lambda x: eval(x))

X = np.array(data['embedding'].tolist())
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = X_train[:500]
y_train = y_train[:500]

print("Enter values for hyperparameters (comma separated)")

n_estimators = list(map(int, input().split(',')))
max_depth_input = input().split(',')
max_depth = [None if x.strip() == 'None' else int(x) for x in max_depth_input]
min_samples_split = list(map(int, input().split(',')))
min_samples_leaf = list(map(int, input().split(',')))
n_iter = int(input())
cv = int(input())

param_dist = {
    'n_estimators': n_estimators,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf
}

rf_model = RandomForestClassifier()

random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_dist,
    n_iter=n_iter,
    cv=cv,
    verbose=2,
    random_state=42,
    n_jobs=1
)

random_search.fit(X_train, y_train)

print(random_search.best_params_)

best_model = random_search.best_estimator_

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
