import pandas as pd
import numpy as np
import shap

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

# Classification models
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Regression models
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression

# Clustering
from sklearn.cluster import AgglomerativeClustering, DBSCAN

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score
)

# ===================== LOAD DATA =====================
data = pd.read_csv("Coherence_bert_cls_embeddings.csv")
data['embedding'] = data['embedding'].apply(lambda x: eval(x))

X = np.array(data['embedding'].tolist())
y = data['label']

# ===================== MENU =====================
print("\nSelect operation:")
print("1. Classification")
print("2. Regression")
print("3. Clustering")
print("4. Hyperparameter Tuning")
print("5. SHAP Analysis")

choice = int(input("Enter choice: "))

# ===================== CLASSIFICATION =====================
if choice == 1:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train[:500]
    y_train = y_train[:500]

    print("Select models (comma separated):")
    print("1.SVM 2.DecisionTree 3.RandomForest 4.AdaBoost 5.NaiveBayes 6.MLP")

    choices = list(map(int, input().split(',')))

    model_map = {
        1: ("SVM", SVC()),
        2: ("DecisionTree", DecisionTreeClassifier()),
        3: ("RandomForest", RandomForestClassifier()),
        4: ("AdaBoost", AdaBoostClassifier()),
        5: ("NaiveBayes", GaussianNB()),
        6: ("MLP", MLPClassifier(max_iter=300))
    }

    results = []

    for c in choices:
        name, model = model_map[c]
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        results.append([
            name,
            accuracy_score(y_train, y_train_pred),
            accuracy_score(y_test, y_test_pred),
            precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
            recall_score(y_test, y_test_pred, average='weighted', zero_division=0),
            f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
        ])

    print(pd.DataFrame(results, columns=["Model","Train Acc","Test Acc","Precision","Recall","F1"]))

# ===================== REGRESSION =====================
elif choice == 2:
    if y.dtype == 'object':
        y = pd.factorize(y)[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train[:500]
    y_train = y_train[:500]

    print("Select regressors:")
    print("1.LinearRegression 2.SVR 3.DecisionTree 4.RandomForest 5.AdaBoost 6.MLP")

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

        results.append([
            name,
            mean_squared_error(y_train, y_train_pred),
            mean_squared_error(y_test, y_test_pred),
            mean_absolute_error(y_train, y_train_pred),
            mean_absolute_error(y_test, y_test_pred),
            r2_score(y_test, y_test_pred)
        ])

    print(pd.DataFrame(results, columns=["Model","Train MSE","Test MSE","Train MAE","Test MAE","R2"]))

# ===================== CLUSTERING =====================
elif choice == 3:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("1.Hierarchical  2.DBSCAN")
    choices = list(map(int, input().split(',')))

    results = []

    for c in choices:
        if c == 1:
            n_clusters = int(input("Clusters: "))
            linkage = input("Linkage: ")

            model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
            labels = model.fit_predict(X_scaled)

        elif c == 2:
            eps = float(input("eps: "))
            min_samples = int(input("min_samples: "))

            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X_scaled)

        score = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else -1
        results.append([c, score])

    print(pd.DataFrame(results, columns=["Method","Silhouette Score"]))

# ===================== HYPERPARAMETER =====================
elif choice == 4:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print("Enter hyperparameters:")
    n_estimators = list(map(int, input().split(',')))
    max_depth = [None if x=='None' else int(x) for x in input().split(',')]
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

    model = RandomForestClassifier()

    search = RandomizedSearchCV(model, param_dist, n_iter=n_iter, cv=cv)
    search.fit(X_train, y_train)

    print("Best Params:", search.best_params_)

# ===================== SHAP =====================
elif choice == 5:
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2)
    X_train = X_train[:200]

    print("1.RandomForest 2.DecisionTree 3.SVM")
    c = int(input())

    if c == 1:
        model = RandomForestClassifier()
    elif c == 2:
        model = DecisionTreeClassifier()
    else:
        model = SVC(probability=True)

    model.fit(X_train, y_train)

    n = int(input("Samples for SHAP: "))
    X_sample = X_train[:n]

    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)

    shap.plots.beeswarm(shap_values)
