import pandas as pd
def jaccard_smc(v1,v2):
    f11 = 0
    f00 = 0
    f10 = 0
    f01 = 0

    for i in range(len(v1)):
        if v1[i] == 1 and v2[i] == 1:
            f11 += 1
        elif v1[i] == 0 and v2[i] == 0:
            f00 += 1
        elif v1[i] == 1 and v2[i] == 0:
            f10 += 1
        elif v1[i] == 0 and v2[i] == 1:
            f01 += 1

        if(f11 + f10 + f01) == 0:
            jc = 0
        else:
            jc = f11 /(f11 + f10 + f01)

        if(f11 + f10 + f01 + f00) == 0:
            smc = 0
        else :
            smc =(f11 + foo)/(f11 + f10 +f01 +f00)
        return jc,smc
def main():
    path=r"C:\Users\heman\OneDrive\Desktop\Lab02\FOLDER\Lab Session Data.xlsx"
    data = pd.read_excel(path, sheet_name="thyroid0387_UCI")
    binary_data = data.select_dtypes(include = "int64").iloc[:2]
    v1 = binary_data.iloc[0].values
    v2 = binary_data.iloc[1].values

    jc,smc = jaccard_smc(v1,v2)

    print("Jaccard Coefficient:",jc)
    print("Simple Mtching Coefficient:",smc)
main()
         
