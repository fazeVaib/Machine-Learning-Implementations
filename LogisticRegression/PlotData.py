import matplotlib.pyplot as plt
import numpy as np

def plotData(x, y):
    pos = []
    neg = []
    for i in range(len(y)):
        if y[i, 0] == 0:
            neg.append([x[i][0], x[i][1]])
        else:
            pos.append([x[i][0], x[i][1]])

    pos = np.array(pos)
    neg = np.array(neg)
    plt.scatter(pos[:, 0], pos[:, 1], marker='^', s=100, alpha=0.5, label="Admitted")
    plt.scatter(neg[:, 0], neg[:, 1], marker='o', s=100, alpha=0.5, label="Not Admitted")
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend()
    plt.show()