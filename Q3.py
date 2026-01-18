import pandas as pd
import numpy as np
def main():
    data=pd.read_excel(r"C:\Users\heman\OneDrive\Desktop\Lab02\FOLDER\Lab Session Data.xlsx",sheet_name="marketing_campaign")
    numeric_col =data.select_dtypes(include=[np.number])
    means = numeric_cols.mean()
    variances =numeric_cols.var()
    print("Mean Values:")
    print(means)
    print("\nVariance values:")
    print(Variance)
    missing=data.isnull().sum()
    print("\n Missing values per column:")
    print(missing)
    main()
