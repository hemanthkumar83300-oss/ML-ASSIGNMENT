import pandas as pd
def normalize(data):
    num = data.select_dtypes(include=["int64" , "float64"])
    return (num - num.min()) / (num.max() - num.min())

def main():
    path = r"C:\Users\heman\OneDrive\Desktop\Lab02\FOLDER\Lab Session Data.xlsx"
    data = pd.read_excel(path, sheet_name="thyroid0387_UCI")
    normalized =normalize(data)
    print(normalized.head())
main()
