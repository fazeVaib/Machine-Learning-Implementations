import numpy as np
import Hypothesis as h
import ComputerCost as cc

def grad(x, y, theta):
    m = x.shape[0]
    grad = np.array([0.0, 0.0])
    for i in range(m):
        grad[0] += (-y[i] + h.hypo(x[i], theta))
        grad[1] += (h.hypo(x[i], theta) - y[i])*x[i]
    return grad

def gradientDescent(x, y, theta, alpha, iter):
    m =x.shape[0]
    e = []
    theta0 = []
    theta1 = []
    for _ in range(iter):
        gradient = grad(x, y, theta)
        theta[0] = theta[0] - (alpha/m)*gradient[0]
        theta[1] = theta[1] - (alpha/m)*gradient[1]
        e.append(cc.cost(x, y, theta))
        theta0.append(theta[0])
        theta1.append(theta[1])
    return e, theta0, theta1, theta