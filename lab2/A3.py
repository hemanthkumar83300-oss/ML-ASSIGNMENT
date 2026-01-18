import pandas as pd
import numpy as np
import time

def manual_mean(values):
    total = 0
    count = 0
    for v in values:
        total += v
        count += 1
    return total / count

def manual_variance(values, mean):
    total = 0
    count =0
    for v in values:
        total += (v-mean) ** 2
        count += 1
    return total /count

def average_time(func , values):
    times = []
    for _ in range(10):
        start = time.time()
        func(values)
        times.append(time.time() - start)
    return sum(times) / len(times)

def main():
    path = r"C:\Users\heman\OneDrive\Desktop\Lab02\FOLDER\Lab Session Data.xlsx"
    data=pd.read_excel(path, sheet_name="IRCTC Stock Price")
    price = data["Price"]
    change =data["Chg%"]
    day = data["Day"]
    np_mean = np.mean(price)
    np_var = np.var(price)
    man_mean = manual_mean(price)
    man_var =manual_variance(price, man_mean)
    manual_time =average_time(manual_mean, price)
    wed_price = data[day =="wed"]["Price"]
    apr_price =data[data["Month"] =="Apr"]["Price"]
    loss_prob =len(list(filter(lambda x :x < 0 ,change ))) /len(change)
    wed_profit =len(data[(day == "Wed") & (change > 0 )])
    wed_total =len(data[day =="Wed"])
    if wed_total == 0 :
        cond_prob = 0
    else:
        cond_prob =wed_profit / wed_total
    print("Numpy Mean :" , np_mean)
    print("Manual Mean:",man_mean)
    print("Numpy Variance:",np_var)
    print("Manual Variance:",man_var)
    print("Average execution time:",manual_time)
    print("Wednesday Sample Mean:",wed_price.mean() if len(wed_price)>0 else 0)
    print("April Sample Mean:",apr_price.mean())
    print("Probability of Loss:",loss_prob)
    print("Conditional Profit Probability :" , cond_prob)
main()      
