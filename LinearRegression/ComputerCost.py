import Hypothesis as h

def cost(x, y, theta):
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        cost += (h.hypo(x[i], theta)- y[i])**2
    return (0.5*cost)/m