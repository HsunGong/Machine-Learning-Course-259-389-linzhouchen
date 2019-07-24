import numpy as np
from numpy import matmul

import matplotlib.pyplot as plt
import os,sys
def report(name, err):
    name = os.path.splitext(sys.argv[0])[0] + " model-" + name

    plt.figure()
    plt.title(name)
    plt.colorbar
    plt.xlabel('$x$')
    plt.ylabel('$err$')

    # f* = 0
    plt.plot(range(len(err)), np.log(err))

    plt.savefig(name)
    plt.show()
    return


# %% Question 1-1

# y = sum(xi * log(xi))
### f* -> dual function(x is m-dim)

from numpy.linalg import pinv

np.random.seed(0)

n = 500
m = 100

# A = np.random.uniform(1e-2, 1, (m, n))
# b = np.random.uniform(1e-2, 1, m)
b = np.load('b.npy')
A = np.load('A.npy')

x0 = np.random.uniform(0, 1, size=m) # in dual function

gradient = lambda x: b - matmul(A, np.exp( - matmul(A.T, x) - 1))

value = lambda x: matmul(b.T, x) + np.sum(np.exp(- matmul(A.T, x) - 1))
# def value(x):
#     ans = matmul(b.T, x) + np.sum(np.exp(- matmul(A.T, x) - 1))
#     print(ans)
#     return ans

def backtrack(x, dx, alpha=0.5, beta=0.99):
    t = 1.0
    # dx = gradient(x)
    g = gradient(x)
    y = value(x)
    g_dx = matmul(g, dx)
    while value(x + t * dx) > y + alpha * t * g_dx:
        t *= beta
    return t


def dualProb(x, maxiter=350):
    ans = [value(x)]  # 0
    i = 0
    while i <= maxiter:
        i += 1
        # print(x)
        dx = - gradient(x)  # direction
        stepSize = backtrack(x, dx)

        x = x + stepSize * dx
        print(i, value(x))
        ans.append(value(x))

    return ans, x


err, ans = dualProb(x0)

# L = f + miu*(Ax-b)
# dL = df(x) + A.T*miu = 0, miu = ans
# df = np.log(x) + 1 = -A.T*miu
# x = exp(-A.T*miu - 1)
x = np.exp(-matmul(A.T, ans) - 1)
f = lambda x: np.sum(x * np.log(x))
print(f(x))

report("Ladmpsap", err[:-1] - err[-1]) # the solution of dual problem
