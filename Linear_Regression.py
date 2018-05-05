import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
import math
df = pd.read_csv('wdbc.data', na_values='none',header=None)    # read the data
yt = [(0 if x == 'M' else 1) for x in df[1]]
data = df.drop([0,1],axis=1)
X = data.as_matrix()                # make X,y be numpy array
y = np.zeros([len(yt), 1])
for i in range(len(yt)):
    y[i][0] = yt[i]

def pre(X):        # normalize the input X
    m,n = X.shape
    maxvalue = np.amax(X, axis = 0)
    minvalue = np.amin(X, axis = 0)
    Xp = (X - minvalue)/(maxvalue-minvalue)
    return Xp
def avg(s):                   # get the average of s. 
    n = 0
    S = 0.0
    for i in range(len(s)):
        if math.isnan(s[i][0]) == False:
            S = S + s[i][0]
            n = n + 1
    if n > 0:
        return S/n
    else:
        return 0
def H(Xn, theta):             # sigmoid equation. 1/(1+e^{WX})
    t = np.matmul(Xn, theta)
    for i in range(len(t)):
        if t[i][0] > 100:
            t[i][0] = 100
        elif t[i][0] < -100:
            t[i][0] = -100
    h = 1/(np.exp(-t)+1)
    return h
def cost(Xn, y, theta, lamb):		# the cost function 
    m,n = Xn.shape
    h = H(Xn, theta)
    s = y*np.log(h) + (1-y)*np.log(1-h)
    #res = -np.sum(s)/m + np.sum(theta*theta)*lamb
    res = -avg(s) + np.sum(theta*theta)*lamb
    return res
def gradient_descent(X,y,alpha, lamb, T):
    X = pre(X)      # first normalize X
    m,n = X.shape
    theta = np.random.uniform(0,0.1,[n+1,1])    # build the initial theta
    costList = []
    b = np.ones(m)
    Xn = np.insert(X, 0, values=b, axis=1)		# add a column with all value 1
    #print(Xn)
    Xt = np.transpose(Xn)
    ite = 0
    while ite < T:
        h = H(Xn, theta)
        theta = theta + alpha*(np.matmul(Xt,y - h)/m - 2*lamb*theta)   # gradient descent's step
        c = cost(Xn, y, theta, lamb)  
        costList.append(c)    # calculate the cost
        ite = ite + 1
    plt.plot(costList)			
    plt.show()			# show the cost curve
    return theta

w = gradient_descent(X,y,0.5,0.1,100)   
print("The gradient curve's weights:\n")
print(w)

# the logistic regression
yn = np.zeros(len(y))
for i in range(len(y)):
    if y[i][0] < 0.5:
        yn[i] = 0
    else:
        yn[i] = 1
lg = LogisticRegression(C=0.05)
lg.fit(X, yn)
print("Logistic Regression's weights\n")
print(lg.coef_)
print(lg.intercept_)



# build the boundary of the curve
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
mpl.rc('figure', figsize=[10,6])
df = pd.read_csv('wdbc.data', header=None)
base_names = ['radius', 'texture', 'perimeter', 'area', 'smooth', 'compact', 'concav', 
                 'conpoints', 'symmetry', 'fracdim']
names = ['m' + name for name in base_names]
names += ['s' + name for name in base_names]
names += ['e' + name for name in base_names]
names = ['id', 'class'] + names
df.columns = names
df['color'] = pd.Series([(0 if x == 'M' else 1) for x in df['class']])
my_color_map = mpl.colors.ListedColormap(['r', 'g'], 'mycolormap')
from sklearn.linear_model import LogisticRegression

c1 = 'mradius'
c2 = 'mtexture'

clf = LogisticRegression()
poly = PolynomialFeatures(3)
NewInput = poly.fit_transform(df[[c1,c2]])       # change the input into term up to degree 3. Get 10 columns
clf.fit(NewInput, df['color'])                   # use the newinput to train the model
plt.scatter(df[c1], df[c2], c = df['color'], cmap=my_color_map)
plt.xlabel(c1)
plt.ylabel(c2)

x = np.linspace(df[c1].min(), df[c1].max(), 1000)
y = np.linspace(df[c2].min(), df[c2].max(), 1000)
xx, yy = np.meshgrid(x,y)
origin = np.hstack((xx.ravel().reshape(-1,1), yy.ravel().reshape(-1,1)))
poly = PolynomialFeatures(3)
ninput = poly.fit_transform(origin)     # build the data from 2 columns to 10 columns.
predictions = clf.predict(ninput)       # get the term to degree 3.
predictions = predictions.reshape(xx.shape)

plt.contour(xx, yy, predictions, [0.0])
plt.show()
