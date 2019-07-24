import matplotlib.pyplot as plt

import numpy as np
from numpy import matmul
from numpy.linalg import norm


import sys, os.path

# print(os.path.splitext(sys.argv[0])[0])
def report(name, err):
    name = os.path.splitext(sys.argv[0])[0] + " model-" + name

    plt.figure()
    plt.title(name)
    plt.colorbar
    plt.xlabel('$x$')
    plt.ylabel('$log(y_b-y_a)$')

    # f* = 0
    plt.plot(range(len(err)), np.log(err))

    plt.savefig(name)
    plt.show()
    return

import numpy.random as rand
np.random.seed(0)

n = 300
m = 200


D = rand.normal(1e-2, 1, (m, n))
b = rand.normal(1e-2, 1, m) # is y in prob

x0 = rand.uniform(-3,3, size=n) # in dual function

value = lambda x: norm(b - D @ x, 2)**2
gradient = lambda x: - 2*matmul(D.T, b - D @ x)


def proj(x): # proj into l1-norm=1
    # ans = x
    index = abs(x) > 1
    x[index] = np.sign(x[index])

    return np.asarray(x)

def backtrack(x, dx, alpha=0.5, beta=0.99):
    t = 1.0
    # dx = gradient(x)
    g = gradient(x)
    y = value(x)
    g_dx = matmul(g, dx)
    while value(x + t * dx) > y + alpha * t * g_dx:
        t *= beta
    return t

from numpy import absolute as abs

def findDir(x):
    # solve: gradient(x0) *x x \in ||||inf<1 -> y
    g = gradient(x)
    
    y = - np.nan_to_num(np.sign(g))
    return y


def FrankWolfeProjection(x, maxiter=20000):
    ans = [value(x)]  # 0
    i = 0
    while i <= maxiter:
        i += 1

        direction = findDir(x) - x 
        # ||∇f(xk)Tdk≤ϵ|| -> end 
        # print(direction)

        # x = x + backtrack(x, direction) * direction
        gamma = 2/(i + 2)
        x = x  + gamma * direction

        # print(norm(x, 1))
        # if (norm(x, 1) > 1):
        #     print('good', norm(x, 1))
        x = proj(x)
        print(i, value(x))
        # print(x.shape)

        ans.append(value(x))

    return ans, x



def FrankWolfe(x, maxiter=1000):
    ans = [value(x)]  # 0
    i = 0
    while i <= maxiter:
        i += 1

        direction = findDir(x) - x 
        # ||∇f(xk)Tdk≤ϵ|| -> end 
        # print(x)
        # if matmul(gradient(x), direction) < 1e-5:
        #     break

        # x = x + backtrack(x, direction) * direction
        gamma = 2/(i + 2)
        x = x  + gamma * direction

        if i % 100 == 0:
            print(i, value(x))
            # print(np.max(direction))
        # print(x.shape)

        ans.append(value(x))

    return ans, x

err, ans = FrankWolfe(x0, 20000)
report("FrankWolfe", err[:10000] - err[-1]) # the solution of dual problem

# err2, ans2 = FrankWolfeProjection(x0, 20000)
# report("FrankWolfeProjection", err2[:10000] - err2[-1]) # the solution of dual problem

# diff = [err[i] - err2[i] for i in range(len(err))]
# diff = np.asarray(diff)
# diff = diff[diff > 0]
# print(len(diff[:50000])/50000)
# report("Compare", np.asarray(diff[:80000]))