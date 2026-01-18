import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def cosine_similarity(v1 ,v2):
    return np.dot(v1 ,v2) /(np.linalg.norm(v1) * np.linalg.norm(v2))

def main():
    path =r"C:\Users\heman\OneDrive\Desktop\Lab02\FOLDER\Lab Session Data.xlsx"
    data =pd.read_excel(path,sheet_name="thyroid0387_UCI").iloc[:20]
    numeric=data.select_dtypes(include=["int64","float64"]).values
    matrix =np.zeros((20,20))
    for i in range(20):
        for j in range(20):
            matrix[i][j] = cosine_similarity(numeric[i],numeric[j])
    sns.heatmap(matrix,annot=True)
    plt.show()
main()

        
                                        
