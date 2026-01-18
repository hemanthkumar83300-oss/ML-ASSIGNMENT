import pandas as pd
import numpy as np

def jaccard(v1, v2):
    f11 = sum((v1 == 1) & (v2 == 1))
    f10 = sum((v1 == 1) & (v2 == 0))
    f01 = sum((v1 == 0) & (v2 == 1))
    denom = f01 + f10 + f11
    if denom == 0:
        return 0
    return f11 / denom

def smc(v1, v2):
    f11 = sum((v1 == 1) & (v2 == 1))
    f00 = sum((v1 == 0) & (v2 == 0))
    f10 = sum((v1 == 1) & (v2 == 0))
    f01 = sum((v1 == 0) & (v2 == 1))
    denom = f00 + f01 + f10 + f11
    if denom == 0:
        return 0
    return (f11 + f00) / denom

def cosine(v1, v2):
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0
    return np.dot(v1, v2) / denom

def main():
    data = pd.read_excel(
        r"C:\Users\heman\OneDrive\Desktop\Lab02\FOLDER\Lab Session Data.xlsx",
        sheet_name="thyroid0387_UCI"
    )

    numeric_data = data.apply(pd.to_numeric, errors="coerce")
    numeric_data = numeric_data.fillna(0)

    v1 = numeric_data.iloc[0].values.astype(float)
    v2 = numeric_data.iloc[1].values.astype(float)

    jc = jaccard(v1, v2)
    smc_val = smc(v1, v2)
    cos = cosine(v1, v2)

    print("Jaccard Coefficient:", jc)
    print("Simple Matching Coefficient:", smc_val)
    print("Cosine Similarity:", cos)

main()
