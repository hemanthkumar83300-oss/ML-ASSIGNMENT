import pandas as pd
def impute(data):
    for col in data.columns:
        if data[col].dtype in ["int64" , "float64"]:
            data[col].fillna(data[col].median() , inplace=True)
        else:
            data[col].fillna(data[col].mode()[0] ,inplace=True)
    return data

def main():
    path = r"C:\Users\heman\OneDrive\Desktop\Lab02\FOLDER\Lab Session Data.xlsx"
    data=pd.read_excel(path, sheet_name="thyroid0387_UCI")
    result = impute(data)
    print(result.isnull().sum())
main()
