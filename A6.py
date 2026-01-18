import numpy as np
import pandas as pd

def cosine_similarity(v1,v2):
    return np.dot(v1,v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
def main():
    path = r"C:\Users\heman\OneDrive\Desktop\Lab02\FOLDER\Lab Session Data.xlsx"
    data = pd.read_excel(path, sheet_name="thyroid0387_UCI")
    numeric = data.select_dtypes(include=["int64","float64"]).iloc[:2].values
    cos = cosine_similarity(numeric[0] , numeric[1])
    print("Cosine Similarities:" ,cos)
main()
