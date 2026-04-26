import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SequentialFeatureSelector

from sklearn.metrics import accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt

import shap
from lime.lime_tabular import LimeTabularExplainer


def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna()
    target_column = "label"
    df = df.drop(columns=["para_id", "sentence"], errors='ignore')
    df["embedding"] = df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=","))
    embedding_df = pd.DataFrame(df["embedding"].tolist())
    X = embedding_df
    y = df[target_column]
    return X, y


def plot_correlation_heatmap(X):
    X_small = X.iloc[:, :20]
    plt.figure(figsize=(10, 6))
    sns.heatmap(X_small.corr(), cmap='coolwarm')
    plt.title("Correlation Heatmap (First 20 Features)")
    plt.show()


def preprocess_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def train_models(X_train, X_test, y_train, y_test):
    results = {}

    lr = LogisticRegression(max_iter=2000)
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    results["Logistic Regression"] = accuracy_score(y_test, lr.predict(X_test))
    results["Random Forest"] = accuracy_score(y_test, rf.predict(X_test))

    return results, lr, rf


def apply_pca(X_train, X_test, variance):
    pca = PCA(n_components=variance)
    return pca.fit_transform(X_train), pca.transform(X_test)


def sequential_feature_selection(X_train, y_train, X_test):
    X_train_small = X_train[:, :30]
    X_test_small = X_test[:, :30]

    model = LogisticRegression(max_iter=1000)

    sfs = SequentialFeatureSelector(
        model,
        n_features_to_select=5,
        direction="forward",
        cv=2,
        n_jobs=1
    )

    sfs.fit(X_train_small, y_train)

    return sfs.transform(X_train_small), sfs.transform(X_test_small)


def lime_explanation(model, X_train, X_test):
    explainer = LimeTabularExplainer(X_train, mode='classification')
    exp = explainer.explain_instance(
        X_test[0],
        model.predict_proba,
        num_features=10
    )
    return exp


def shap_explanation(model, X_train):
    explainer = shap.Explainer(model, X_train[:100])
    shap_values = explainer(X_train[:10])
    return shap_values


if __name__ == "__main__":

    FILE_PATH = r"C:\Users\heman\Downloads\LAB10\LAB10\Coherence_bert_cls_embeddings.csv"

    X, y = load_data(FILE_PATH)

    plot_correlation_heatmap(X)

    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    original_results, lr_model, rf_model = train_models(
        X_train, X_test, y_train, y_test
    )
    print("\nOriginal:", original_results)

    X_train_99, X_test_99 = apply_pca(X_train, X_test, 0.99)
    res_99, _, _ = train_models(X_train_99, X_test_99, y_train, y_test)
    print("\nPCA 99%:", res_99)

    X_train_95, X_test_95 = apply_pca(X_train, X_test, 0.95)
    res_95, _, _ = train_models(X_train_95, X_test_95, y_train, y_test)
    print("\nPCA 95%:", res_95)

    X_train_sfs, X_test_sfs = sequential_feature_selection(
        X_train, y_train, X_test
    )
    res_sfs, _, _ = train_models(X_train_sfs, X_test_sfs, y_train, y_test)
    print("\nSFS:", res_sfs)

    lime_exp = lime_explanation(lr_model, X_train, X_test)
    print("\nLIME:", lime_exp.as_list())

    shap_values = shap_explanation(rf_model, X_train)

    try:
        shap.plots.beeswarm(shap_values[..., 1])
    except:
        shap.summary_plot(shap_values, X_train[:10])
