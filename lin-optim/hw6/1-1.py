from math import *
import matplotlib.pyplot as plt

import numpy as np
from numpy import matmul

import os, sys
def report(method, err):
    name = os.path.splitext(sys.argv[0])[0] + " model-" + method.__name__

    plt.figure()
    plt.title(name)
    plt.colorbar
    plt.xlabel('$x$')
    plt.ylabel('$log-(y-y*)$')

    # f* = 0
    plt.plot(range(len(err)), (err))

    plt.savefig(name)
    plt.show()
    return


# %% Question 1-1


from numpy.linalg import pinv

np.random.seed(2019)

n = 500
m = 100
# x0 = np.absolute(np.random.uniform(1e-2, 1, size=n))
x0 = np.abs(np.random.rand(n))
A = np.random.random((m, n))

# A = np.random.uniform(0, 1, (m, n))
b = A @ x0
np.save('b', b)
np.save('A', A)
P = np.identity(n) - pinv(A) @ A

# print(b)

# print((pinv(A) == A.T @ (A @ A.T).I).all())

# x0 = matmul(P, x0) + matmul(pinv(A), b) # P(x0) \in Omega

# print([(p < 0) for p in x0])

# y = sum(xi * log(xi))
value = lambda x :  np.sum(x * np.log(x))
gradient = lambda x :  1 + np.log(x)


def backtrack(x, dx, alpha=0.9, beta=0.9):
    t = 1.0
    # dx = gradient(x)
    g = gradient(x)
    f = value(x)
    g_dx = matmul(g, dx)
    while value(x + t * dx) > f + alpha * t * g_dx:
        t *= beta
    return t

def projectionGradient(method, x, maxiter=90):
    ans = [value(x)]  # 0
    i = 0
    while i <= maxiter:
        i += 1
        dx = -gradient(x)  # direction
        stepSize = backtrack(x, dx)


        x = x + stepSize * P @ dx
        if (x <= 0).any():
            raise "Error"
        ans.append(value(x))
        print(i, stepSize, value(x))

    return ans, x


err, ans = projectionGradient(projectionGradient, x0)
report(projectionGradient, err[:-1] - err[-1])
