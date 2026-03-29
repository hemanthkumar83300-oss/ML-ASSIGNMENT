import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.decomposition import PCA


# -------------------- COMMON FUNCTIONS --------------------

def binning(data, bins=4):
    mn = min(data)
    mx = max(data)
    w = (mx - mn) / bins
    res = []
    for v in data:
        idx = int((v - mn) / w)
        if idx == bins:
            idx -= 1
        res.append(idx)
    return res


def entropy(data):
    total = len(data)
    freq = {}
    for v in data:
        freq[v] = freq.get(v, 0) + 1
    h = 0
    for c in freq.values():
        p = c / total
        h += -p * math.log2(p)
    return h


def gini(data):
    total = len(data)
    freq = {}
    for v in data:
        freq[v] = freq.get(v, 0) + 1
    g = 1
    for c in freq.values():
        p = c / total
        g -= p * p
    return g


def info_gain(feature, target):
    total_entropy = entropy(target)
    values = set(feature)
    w_entropy = 0
    for v in values:
        subset = []
        for i in range(len(feature)):
            if feature[i] == v:
                subset.append(target[i])
        w_entropy += (len(subset)/len(target)) * entropy(subset)
    return total_entropy - w_entropy


# -------------------- MAIN PROGRAM --------------------

file = input("Enter CSV file path: ").strip()
df = pd.read_csv(file)

print("\nColumns:", df.columns)

print("\nChoose Operation:")
print("1 → Entropy (with binning)")
print("2 → Gini Index")
print("3 → Information Gain (Root Feature)")
print("4 → Binning only")
print("5 → Decision Tree (Embeddings)")
print("6 → Decision Tree Visualization")
print("7 → Decision Boundary (PCA + Plot)")

choice = int(input("Enter choice: "))


# -------------------- OPERATIONS --------------------

if choice == 1:
    target = input("Enter target column: ")
    y = df[target].values
    y = binning(y)
    print("Entropy:", entropy(y))


elif choice == 2:
    target = input("Enter target column: ")
    y = df[target].values
    print("Gini Index:", gini(y))


elif choice == 3:
    target = input("Enter target column: ")
    y = df[target].values

    cols = [c for c in df.columns if c != target]
    gains = []

    for c in cols:
        g = info_gain(df[c].values, y)
        gains.append(g)
        print(c, "Gain:", g)

    print("Root Feature:", cols[gains.index(max(gains))])


elif choice == 4:
    col = input("Enter column to bin: ")
    bins = int(input("Enter bins (0 for default): "))

    data = df[col].values

    if bins == 0:
        result = binning(data)
    else:
        result = binning(data, bins)

    print(result)


elif choice == 5:
    target = input("Enter target column: ")
    emb_col = input("Enter embedding column name: ")

    X = df[emb_col].apply(lambda x: np.array(eval(x)))
    X = np.vstack(X)
    y = df[target].values

    model = DecisionTreeClassifier(criterion="entropy")
    model.fit(X, y)

    print("Decision Tree Built")


elif choice == 6:
    target = input("Enter target column: ")
    emb_col = input("Enter embedding column name: ")

    X = df[emb_col].apply(lambda x: np.array(eval(x)))
    X = np.vstack(X)
    y = df[target].values

    model = DecisionTreeClassifier(criterion="entropy")
    model.fit(X, y)

    plt.figure(figsize=(12, 6))
    plot_tree(model, filled=True, max_depth=3)
    plt.show()


elif choice == 7:
    target = input("Enter target column: ")
    emb_col = input("Enter embedding column name: ")

    X = df[emb_col].apply(lambda x: np.array(eval(x)))
    X = np.vstack(X)
    y = df[target].values

    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)

    model = DecisionTreeClassifier(criterion="entropy")
    model.fit(X2, y)

    x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
    y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.5)
    plt.scatter(X2[:, 0], X2[:, 1], c=y)
    plt.show()


else:
    print("Invalid choice") 