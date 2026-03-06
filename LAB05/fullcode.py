import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

data = pd.read_csv(r"C:\Users\heman\Downloads\LAB05 (1)\LAB05\projectds.csv")


data["embedding"] = data["embedding"].apply(ast.literal_eval)

X_full = pd.DataFrame(data["embedding"].to_list())


y = data["label"]

print("\n===== LINEAR REGRESSION (Single Attribute) =====")

X_single = X_full.iloc[:, [0]]

X_train, X_test, y_train, y_test = train_test_split(
    X_single, y, test_size=0.2, random_state=42
)

reg = LinearRegression().fit(X_train, y_train)

y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)

mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
mape_train = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
r2_train = r2_score(y_train, y_train_pred)


mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
mape_test = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
r2_test = r2_score(y_test, y_test_pred)

print("TRAIN METRICS")
print("MSE:", mse_train)
print("RMSE:", rmse_train)
print("MAPE:", mape_train)
print("R2:", r2_train)

print("\nTEST METRICS")
print("MSE:", mse_test)
print("RMSE:", rmse_test)
print("MAPE:", mape_test)
print("R2:", r2_test)



print("\n===== LINEAR REGRESSION (All Attributes) =====")

X = X_full

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

reg = LinearRegression().fit(X_train, y_train)

y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)


mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
mape_train = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
r2_train = r2_score(y_train, y_train_pred)


mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
mape_test = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
r2_test = r2_score(y_test, y_test_pred)

print("TRAIN METRICS (All Attributes)")
print("MSE:", mse_train)
print("RMSE:", rmse_train)
print("MAPE:", mape_train)
print("R2:", r2_train)

print("\nTEST METRICS (All Attributes)")
print("MSE:", mse_test)
print("RMSE:", rmse_test)
print("MAPE:", mape_test)
print("R2:", r2_test)



print("\n===== K-MEANS CLUSTERING =====")

kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto")
kmeans.fit(X)

print("Cluster Labels (first 10):")
print(kmeans.labels_[:10])

print("\nCluster Centers Shape:")
print(kmeans.cluster_centers_.shape)


labels = kmeans.labels_

sil_score = silhouette_score(X, labels)
ch_score = calinski_harabasz_score(X, labels)
db_score = davies_bouldin_score(X, labels)

print("\nCluster Evaluation Metrics")
print("Silhouette Score:", sil_score)
print("Calinski-Harabasz Score:", ch_score)
print("Davies-Bouldin Index:", db_score)


k_values = range(2, 11)

sil_scores = []
ch_scores = []
db_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(X)

    labels = kmeans.labels_

    sil_scores.append(silhouette_score(X, labels))
    ch_scores.append(calinski_harabasz_score(X, labels))
    db_scores.append(davies_bouldin_score(X, labels))

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.plot(k_values, sil_scores, marker='o')
plt.title("Silhouette Score vs k")
plt.xlabel("k")
plt.ylabel("Silhouette Score")

plt.subplot(1,3,2)
plt.plot(k_values, ch_scores, marker='o')
plt.title("CH Score vs k")
plt.xlabel("k")
plt.ylabel("CH Score")

plt.subplot(1,3,3)
plt.plot(k_values, db_scores, marker='o')
plt.title("DB Index vs k")
plt.xlabel("k")
plt.ylabel("DB Index")

plt.tight_layout()
plt.show()

distortions = []

for k in range(2, 20):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(X)
    distortions.append(kmeans.inertia_)

plt.plot(range(2, 20), distortions, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.show()