# In[]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PlotData
import scipy.optimize as op

# In[]
df = pd.read_csv("ex2data1.txt", names=['Exam 1', 'Exam 2', 'Admitted'])
x = df.iloc[:,0:2].values
y = df.iloc[:,2:3].values

# In[]
PlotData.plotData(x,y)
theta = np.zeros([1,3])
ones = np.ones([x.shape[0],1])
X = np.concatenate((ones,x),axis=1)

# In[]
def sigmoid(z):
    return  1/(1 + np.exp(-z))

def cost(x, y, theta):
    a = y * np.log(sigmoid(x @ theta.T))
    b = (1-y) * np.log(1 - sigmoid(x @ theta.T))
    return np.sum(-(a+b)/len(x))

def gradient(x, y, theta, alpha, iters):
    costlist = np.zeros(iters)
    for i in range(iters):
        theta = theta - (alpha/len(x))*(np.sum(x * (sigmoid(x @ theta.T) - y)))
        costlist[i] = cost(x, y, theta)
    return theta, costlist

# In[]
print(cost(X,y,theta))
alpha = 0.000001
iters = 2000
theta, costlist = gradient(X, y, theta, alpha, iters)
newCost = cost(x=X, y=y, theta=theta)
print(theta)
print(newCost)

# newtheta = op.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X,y))
# cost(X, y, newtheta[0])

# In[]
plt.plot(np.arange(iters), costlist, 'g')
plt.xlabel("Iterations")
plt.ylabel('Cost')
plt.show()

# print(sigmoid([[1, 45, 85]] @ theta.T))