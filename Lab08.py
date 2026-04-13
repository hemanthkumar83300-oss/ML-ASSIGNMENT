import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import os

def get_data(path):

    if not os.path.exists(path):
        print("File not found")
        exit()

    try:
        df = pd.read_excel(path, engine='openpyxl')
    except:
        df = pd.read_csv(path)

    
    df.replace({'Yes':1, 'No':0}, inplace=True)

   
    df = df.select_dtypes(include=[np.number])

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    return X, y

def calc_sum(x, w, b):
    return np.dot(x, w) + b

def step_fun(x):
    return 1 if x >= 0 else 0

def bipolar_fun(x):
    return 1 if x >= 0 else -1

def sigmoid_fun(x):
    return 1 / (1 + np.exp(-x))

def relu_fun(x):
    return max(0, x)

def find_error(t, o):
    return t - o


def train_model(X, y, w, b, lr, act):

    err_list = []

    for ep in range(1000):
        total_err = 0

        for i in range(len(X)):
            y_in = calc_sum(X[i], w, b)
            y_out = act(y_in)

            err = find_error(y[i], y_out)

            # update rule
            w = w + lr * err * X[i]
            b = b + lr * err

            total_err += err**2

        err_list.append(total_err)

        if total_err <= 0.002:
            break

    return w, b, err_list, ep

def pseudo_method(X, y):
    X1 = np.c_[np.ones(len(X)), X]
    return np.linalg.pinv(X1).dot(y)

def backprop_train(X, y):

    w1 = np.random.rand(X.shape[1], 4)
    w2 = np.random.rand(4, 1)

    for ep in range(2000):
        total = 0

        for i in range(len(X)):
            x = X[i].reshape(1,-1)

            h = sigmoid_fun(np.dot(x, w1))
            o = sigmoid_fun(np.dot(h, w2))

            e = y[i] - o
            total += e**2

            d2 = e * o * (1-o)
            d1 = d2.dot(w2.T) * h * (1-h)

            w2 += 0.05 * h.T.dot(d2)
            w1 += 0.05 * x.T.dot(d1)

        if total <= 0.002:
            break

    return total

def two_output_train(X, y):

    w = np.random.rand(X.shape[1], 2)
    b = np.random.rand(2)

    for ep in range(1000):
        total = 0

        for i in range(len(X)):
            net = np.dot(X[i], w) + b
            out = sigmoid_fun(net)

            err = y[i] - out

            w += 0.05 * np.outer(X[i], err)
            b += 0.05 * err

            total += np.sum(err**2)

        if total <= 0.002:
            break

    return w

if __name__ == "__main__":

    
    file_path = r"C:\Users\heman\Downloads\Lab08\Lab08\Coherence_bert_cls_embeddings.csv"

    X_data, y_data = get_data(file_path)

   
    sc = StandardScaler()
    X_data = sc.fit_transform(X_data)

    
    X_and = np.array([[0,0],[0,1],[1,0],[1,1]])
    y_and = np.array([0,0,0,1])

   
    X_xor = np.array([[0,0],[0,1],[1,0],[1,1]])
    y_xor = np.array([0,1,1,0])

    w = np.array([0.2, -0.75])
    b = 10

    
    _,_,err,ep = train_model(X_and,y_and,w.copy(),b,0.05,step_fun)
    print("A2:", ep)

    plt.plot(err)
    plt.title("Error graph")
    plt.show()

    
    for f in [bipolar_fun, sigmoid_fun, relu_fun]:
        _,_,_,ep = train_model(X_and,y_and,w.copy(),b,0.05,f)
        print("A3",f.__name__,ep)

    
    rates = np.arange(0.1,1.1,0.1)
    ep_list = []

    for r in rates:
        _,_,_,ep = train_model(X_and,y_and,w.copy(),b,r,step_fun)
        ep_list.append(ep)

    plt.plot(rates,ep_list)
    plt.title("Learning rate vs epochs")
    plt.show()

   
    _,_,_,ep = train_model(X_xor,y_xor,w.copy(),b,0.05,step_fun)
    print("A5:", ep)

    
    w_d = np.random.rand(X_data.shape[1])
    b_d = np.random.rand()

    _,_,_,ep = train_model(X_data,y_data,w_d,b_d,0.01,sigmoid_fun)
    print("A6:", ep)

    
    print("A7:", pseudo_method(X_and,y_and))

   
    print("A8:", backprop_train(X_and,y_and))
    print("A9:", backprop_train(X_xor,y_xor))

    
    y_two = np.array([[1,0],[1,0],[1,0],[0,1]])
    two_output_train(X_and,y_two)
    print("A10 done")

    
    m1 = MLPClassifier(hidden_layer_sizes=(4,), max_iter=5000)
    m1.fit(X_and,y_and)

    m2 = MLPClassifier(hidden_layer_sizes=(4,), max_iter=5000)
    m2.fit(X_xor,y_xor)

    print("A11 AND:", m1.predict(X_and))
    print("A11 XOR:", m2.predict(X_xor))
    m3 = MLPClassifier(hidden_layer_sizes=(5,), max_iter=5000)
    m3.fit(X_data,y_data)

    print("A12:", m3.predict(X_data))
