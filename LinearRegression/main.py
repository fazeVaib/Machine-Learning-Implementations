import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Hypothesis as h
from sklearn import linear_model
import ComputerCost as cc
import GradientDescent as gd
from mpl_toolkits.mplot3d import Axes3D
# In[]
df = pd.read_csv('ex1data1.txt', delimiter=',', header=None, names=['x','y'])
m = df.shape[0]
x = df.x.values
y = df.y.values

# x = pd.read_csv('linearX.csv').values
# y = pd.read_csv('linearY.csv').values
# m = x.shape[0]
# x = x.reshape((m,))
# y = y.reshape((m,))
# x = (x - x.mean())/x.std()

# In[]
# plt.scatter(x=x, y=y, marker="x")
# plt.show()
theta = np.array([0.0, 0.0])

# In[]
J = cc.cost(x, y, theta=theta)
alpha = 0.01
iterations = 2500
e, theta0, theta1, theta = gd.gradientDescent(x, y, theta, alpha, iterations)
print(theta)
plt.scatter(x, y,)
plt.plot(x, h.hypo(x,theta), color='red')
plt.show()

# In[]
plt.plot(e)
plt.show()