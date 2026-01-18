import pandas as pd
import numpy as np

def load_purchase_data(path):
    return pd.read_excel(path, sheet_name="Purchase data")

def get_X_y(data):
    X = data[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
    y = data["Payment (Rs)"].values.reshape(-1, 1)
    return X, y

def get_rank(matrix):
    return np.linalg.matrix_rank(matrix)

def get_cost_using_pinv(X, y):
    return np.linalg.pinv(X) @ y

def main():
    path = r"C:\Users\navya\OneDrive\Desktop\FOLDER\Lab Session Data.xlsx"
    
    data = load_purchase_data(path)
    X, y = get_X_y(data)

    rank = get_rank(X)
    cost = get_cost_using_pinv(X, y)

    print("Dimensionality:", X.shape[1])
    print("Number of Vectors:", X.shape[0])
    print("Rank of Feature Matrix:", rank)
    print("Cost of Candies:", cost[0][0])
    print("Cost of Mangoes:", cost[1][0])
    print("Cost of Milk Packets:", cost[2][0])

main()
