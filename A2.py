import pandas as pd
def load_purchase_data(path):
    return pd.read_excel(path, sheet_name="Purchase data")
def classify_customers(data):
    status_list = []
    for payment in data["Payment (Rs)"]:
        if payment >200:
            status_list.append("RICH")
        else:
            status_list.append("POOR")
    data["Status"] = status_list
    return data
def main():
    path = r"C:\Users\heman\OneDrive\Desktop\Lab02\FOLDER\Lab Session Data.xlsx"
    data =load_purchase_data(path)
    result = classify_customers(data)

    print("Customers Classification :")
    print(result[["Payment (Rs)" , "Status"]])
main()
