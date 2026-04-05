import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

data = pd.read_csv("Coherence_bert_cls_embeddings.csv")

data['embedding'] = data['embedding'].apply(lambda x: eval(x))

X = np.array(data['embedding'].tolist())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Select clustering methods (comma separated numbers):")
print("1.Hierarchical  2.DBSCAN")

choices = list(map(int, input().split(',')))

results = []

for c in choices:
    
    if c == 1:
        n_clusters = int(input("Enter number of clusters: "))
        linkage = input("Enter linkage (ward/complete/average): ")
        
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        labels = model.fit_predict(X_scaled)
        
        if len(set(labels)) > 1:
            score = silhouette_score(X_scaled, labels)
        else:
            score = -1
        
        results.append(["Hierarchical", n_clusters, linkage, score])
    
    elif c == 2:
        eps = float(input("Enter eps value (e.g., 0.5): "))
        min_samples = int(input("Enter min_samples: "))
        
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X_scaled)
        
        if len(set(labels)) > 1:
            score = silhouette_score(X_scaled, labels)
        else:
            score = -1
        
        results.append(["DBSCAN", eps, min_samples, score])

results_df = pd.DataFrame(results, columns=[
    "Algorithm", "Param1", "Param2", "Silhouette Score"
])

print(results_df)
