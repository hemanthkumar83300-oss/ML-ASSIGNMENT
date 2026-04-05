import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

data = pd.read_csv("Coherence_bert_cls_embeddings.csv")

data['embedding'] = data['embedding'].apply(lambda x: eval(x))

X = np.array(data['embedding'].tolist())
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = X_train[:500]
y_train = y_train[:500]

print("Select models (comma separated numbers):")
print("1.SVM  2.DecisionTree  3.RandomForest  4.AdaBoost  5.NaiveBayes  6.MLP")

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

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

    results.append([name, train_acc, test_acc, precision, recall, f1])

results_df = pd.DataFrame(results, columns=[
    "Model", "Train Accuracy", "Test Accuracy", "Precision", "Recall", "F1 Score"
])

print(results_df)
