import pandas as pd
import numpy as np
def load_data(path):
    return pd.read_excel(path,sheet_name="Purchase data")
def get_xy(data):
    X=data[["Candies (#)" , "Mangoes (Kg)","Milk Packets (#)"]].values
    y=data["Payment (Rs)"].values.reshape(-1,1)
    return X,y
def cost_using_pinv(X,y):
    return np.linalg.pinv(X) @ y
def main():
    path=r"C:\Users\heman\OneDrive\Desktop\Lab02\FOLDER\Lab Session Data.xlsx"
    data=load_data(path)

    X,y =get_xy(data)
    X1= X[:3,:]
    y1=y[:3]

    X2=X[3:6,:]
    y2=y[3:6]

    cost_full =cost_using_pinv(X,y)
    cost_1 =cost_using_pinv(X1,y1)
    cost_2 = cost_using_pinv(X2,y2)
    print("Cost using full data:",cost_full.flatten())
    print("cost using square matrix 1:",cost_1.flatten())
    print("Cost using square matrix 2:",cost_2.flatten())
main()
    
