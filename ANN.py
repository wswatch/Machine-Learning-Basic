import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

class ANN:
    # label is the number of classes in y
    def  __init__(self, num_of_layer, size_of_layer, label):
        self.intern_num = num_of_layer
        self.intern_size = size_of_layer
        self.label_num = label
    # set initial value of w and bias   
    def setW(self, X_in, y_out):
        self.w = []
        middle = self.intern_size
        val = math.sqrt(6.0/(X_in+ middle))
        w0 = np.random.uniform(-val,val,[X_in, middle])
        self.w.append(w0)
        for i in range(self.intern_num-1):
            val = math.sqrt(3.0/ middle)
            wi = np.random.uniform(-val, val, [middle, middle])
            self.w.append(wi)
        val = math.sqrt(6.0/(middle + y_out))
        we = np.random.uniform(-val, val, [middle, y_out])
        self.w.append(we)
        
        bias = []   # initialize the bias
        for i in range(self.intern_num):
            bias.append(np.zeros(self.intern_size))
        bias.append(np.zeros(y_out))
        self.bias = bias
    # sigmoid action function    
    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))
    def grad_sigmoid(self, x):
        return x*(1-x)
        #return sigmoid(x) * (1 - sigmoid(x))
    # loss function
    def squared(self, y, yhat):
        return np.sum((y - yhat)**2) / 2.0

    def grad_squared(self, y, yhat):
        return yhat - y
    def pre(self, X):      # do normalize
        X = X / 255.0
        return X
    def fit(self, X, y, alpha, t):
        draw = t / 100
        
        draw_x = []
        acurrate_plot = []
        
        oX = X
        X = self.pre(X)
        m,n = X.shape
        n_in = n
        n_out = self.intern_size
        self.setW(n, self.label_num)
        w = self.w
        bias = self.bias
        a = np.zeros([self.intern_size, self.intern_size])
        for io in range(t):
            i = io % m
            xi = X[i]
            xi.shape = (1, n)
            # forward propagation
            a[0] = self.sigmoid(np.matmul(xi, w[0]) + bias[0])  
            for i in range(1, self.intern_num):
                a[i] = self.sigmoid(np.matmul(a[i-1], w[i]) + bias[i])
                
            ny = self.sigmoid(np.matmul(a[self.intern_num-1], w[self.intern_num]) + bias[self.intern_num])
            yo = np.zeros(self.label_num)
            
            yo[y[i]] = 1.0
            # backward propagation
            error = self.squared(yo, ny)
            
            delta = (yo-ny)* ny * (1 - ny)
            delta.shape = (1, self.label_num)
            j = self.intern_num
            
            while j > 0:
                a_j_T = a[j-1]
                a_j_T.shape = (1, self.intern_size)   # get a[j-1].T
                a_j_T = a_j_T.transpose()
                
                a_pre = np.matmul(delta, w[j].T)
                temp = a_pre * a[j-1] * (1 - a[j-1])   # get dLosss/da_j
                
                w[j] = w[j] + alpha * np.matmul(a_j_T, delta)  # update w,bias
                bias[j] = bias[j] + alpha * delta
                
                delta = temp
                delta.shape = (1, temp.size)
                j = j - 1
            # n is the feature of X_in
            x_j_T = xi.T
            w[0] = w[0] + alpha * np.matmul(x_j_T, delta)
            bias[0] = bias[0] + alpha * delta
            if io % draw == 0:
                draw_x.append(io)
                self.w = w
                self.bias = bias
                pred = self.predict(oX)
                acurrate_plot.append(check(pred, y))
        self.w = w
        self.bias = bias
        plt.plot(draw_x, acurrate_plot)
        plt.show()
    def print(self):
        n = len(self.w)
        for i in range(n):
            print("The w%d are:" %(i))
            print(self.w[i])
            print("The bias%d are:" %(i))
            print(self.bias)
            
    def predict(self, T):    # 1*n
        a = self.pre(T)
        w = self.w
        bias = self.bias
        for i in range(self.intern_num + 1):
            a = np.matmul(a, w[i]) + bias[i]
            a = self.sigmoid(a)
        return a
def check(ny, y):    # get accurate rate
    fin = []
    for r in ny:
        t = r.argmax()
        fin.append(t)
    success = 0.0
    n = len(fin)
    for i in range(n):
        if fin[i] == y[i]:
            success = success + 1
    accurate = success/n
    #print(fin)
    return accurate

from time import time
df = pd.read_csv('train.csv')
start_time = time()

ann = ANN(1, 80, 10)
train = df.head(35000)
test = df.tail(7000)
x_train = train.drop(['label'], axis=1).as_matrix()
y_train = train['label'].as_matrix()

x_test = test.drop(['label'], axis=1).as_matrix()
y_test = test['label'].as_matrix()

ann.fit(x_train,y_train,0.01, 100000)

train_predict = ann.predict(x_train)
a = check(train_predict, y_train)
print("The training accurate is %f" %(a))
test_predict = ann.predict(x_test)
b = check(test_predict, y_test)
print("The testing accurate is %f" %(b))
print('finish')

end_time = time()
time_taken = end_time - start_time
print('The time cost is %f s' %(time_taken))