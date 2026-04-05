import pandas as pd
import numpy as np
import shap

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("Coherence_bert_cls_embeddings.csv")

data['embedding'] = data['embedding'].apply(lambda x: eval(x))

X = np.array(data['embedding'].tolist())
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = X_train[:200]

print("Select model for SHAP:")
print("1.RandomForest  2.DecisionTree  3.SVM")

choice = int(input())

if choice == 1:
    model = RandomForestClassifier()
elif choice == 2:
    model = DecisionTreeClassifier()
elif choice == 3:
    model = SVC(probability=True)

model.fit(X_train, y_train)

print("Enter number of samples for SHAP analysis:")
n_samples = int(input())

X_sample = X_train[:n_samples]

explainer = shap.Explainer(model, X_sample)
shap_values = explainer(X_sample)

shap.plots.beeswarm(shap_values)
