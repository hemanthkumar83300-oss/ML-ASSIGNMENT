import pandas as pd
def analyze_data(data):
    return data.dtypes, data.isnull().sum(),data.describe()
def main():
    path =r"C:\Users\heman\OneDrive\Desktop\Lab02\FOLDER\Lab Session Data.xlsx"
    data = pd.read_excel(path, sheet_name="thyroid0387_UCI")
    dtypes ,missing,stats = analyze_data(data)
    print("Datatypes:\n" ,dtypes)
    print("\n,issing Values:\n",missing)
    print("\nStatistics:\n",stats)
main()
