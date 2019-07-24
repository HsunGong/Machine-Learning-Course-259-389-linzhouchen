import numpy as np

from numpy import matmul
from numpy.linalg import norm
np.random.seed(0)

n = 300
m = 200

x0 = np.random.normal(scale=10, size=n)
A = np.random.normal(scale=10, size=(m, n))
b = np.random.normal(scale=10, size=m)
# print(x0.shape, A.shape, b.shape)


# ans = []


def callback(xk):
    ans.append(0.5*fun(xk))
    print(ans[-1])
    return False

import matplotlib.pyplot as plt
import os,sys
def report(name, err):
    name = os.path.splitext(sys.argv[0])[0] + " model-" + name

    plt.figure()
    plt.title(name)
    plt.colorbar
    plt.xlabel('$x$')
    plt.ylabel('$log-(y - y*)$')

    # f* = 0
    plt.plot(range(len(err)), np.log(err))

    plt.savefig(name)
    plt.show()
    return


# sci()
# report('sci', ans - ans[-1])

gamma = 0.1
fun = lambda x: norm(x, 2)
value = lambda x: 0.5* fun(x) + gamma * norm(matmul(A, x) - b, 2)**2
gradient = lambda x: 0.5/fun(x) * x + gamma * matmul(A.T, A @ x - b)


## %% penalty
def penalty(x, maxiter=1000):
    ans = [value(x)]  # 0
    i = 0
    while i <= maxiter:
        i += 1
        # print(x)
        dx = - gradient(x)  # direction
        stepSize = backtrack(x, dx)
        if (norm(dx, 2) < 1e-8):
            break
        print(norm(dx, 2))

        x = x + stepSize * dx
        print(i, value(x))
        ans.append(value(x))

    return ans, x



def backtrack(x, dx, gamma=0.5, beta=0.99):
    t = 1.0
    # dx = gradient(x)
    g = gradient(x)
    y = value(x)
    g_dx = matmul(g, dx)
    while value(x + t * dx) > y + gamma * t * g_dx:
        t *= beta
    return t



err, ans = penalty(x0)
report("penalty", err[:800] - err[-1])