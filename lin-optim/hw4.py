from math import *
import matplotlib.pyplot as plt
from pylab import *

import numpy as np
import tensorflow as tf
from numpy.linalg import inv
import numpy.linalg as la

import scipy.optimize as spo

from numpy import dot, matmul
from numpy.linalg import pinv


def hessian(x):  # n*n
    ans = np.zeros((50, 50))

    ans[1:50, 1:50] += np.diag(np.ones((49,)) * 200.0)  # 1-50 only
    ans[0:49, 0:49] += np.diag(np.ones((49,)) * 2.0)  # 0-49 only
    ans[0:49, 0:49] += np.diag(x[0:49] * x[0:49] * 1200.0)  # 0-49 only
    ans[0:49, 0:49] += np.diag(-400 * x[1:50])  # 0-49 only

    ans[0:49, 1:50] += np.diag(-400.0 * x[0:49])
    ans[1:50, 0:49] += np.diag(-400.0 * x[0:49])
    return np.matrix(ans)


def gradient(x):  # n*1
    dx = np.zeros((50))
    dx[1:50] += 200 * (x[1:50] - x[0:49] * x[0:49])
    dx[0:49] += 200 * (x[1:50] -
                       x[0:49] * x[0:49]) * (-2 * x[0:49]) + 2 * (x[0:49] - 1)
    return dx


def r(x):  # m*1, f = r.T*r
    r = np.zeros(98)
    r[0:49] += 10 * (x[1:50] - x[0:49] * x[0:49])
    r[49:98] += 1 - x[0:49]
    return r


def jacobian_r(x):  # m*n, j(r(x))
    j = np.matrix(np.zeros((98, 50)))
    j[49:98, 0:49] += -np.diag([1] * 49)
    j[0:49, 1:50] += np.diag([10] * 49)
    j[0:49, 0:49] += -20 * np.diag(x[0:49])
    return j


def value(x):
    ans = 0
    for i in range(49):
        ans += 100 * (x[i + 1] - x[i]**2)**2 + (1 - x[i])**2
    return ans


def report(method, err):
    name = "model-" + method.__name__

    plt.figure()
    title(name, color="b")
    plt.xlabel('$x$')
    plt.ylabel('$y$')

    # f* = 0
    plt.plot(range(len(err)), [log10(i - 0) for i in err])

    plt.savefig(name)
    plt.show()
    return


def trinary(x, dx, left, right, err=1e-5):
    lv = value(x)
    rv = value(x + dx)
    while abs(right - left) >= err:
        t2 = (left*2 + right) / 3
        t3 = (left + right*2) / 3
        f2 = value(x + t2*dx)
        f3 = value(x + t3*dx)

        if f2 < f3:
            right = t3
            rv = f3
        else:
            left = t2
            lv = f2
        
    return (left + right)/2


def exactLine(x):
    g = - gradient(x)
    return trinary(x, g, 0, 1) * g


def backtrack(x, dx, alpha=0.5, beta=0.99):
    t = 1.0
    # dx = gradient(x)
    g = gradient(x)
    f = value(x)
    g_dx = dot(g, dx)
    while value(x + t * dx) > f + alpha * t * g_dx:
        t *= beta
    return t


def backtrackLine(x):
    g = - gradient(x)
    return backtrack(x, g, 0.1, 0.9) * g


def getAlpha(x, s, c1=0.001):
    alpha = spo.line_search(value, gradient, x, s, c1=c1, c2=0.99)[0]
    if alpha is None:
        alpha = 1
    return alpha


def gradientDescent(x):
    return -gradient(x)


def classicNewton(x):
    return matmul(hessian(x).I, -gradient(x)).A1


def dampedNewton(x):
    s = classicNewton(x)
    return getAlpha(x, s, 1.2) * s


def gaussNewton(x):
    j = jacobian_r(x)
    return (-matmul(pinv(j), r(x))).A1


def levenbergMarquardt(x):
    u = 1e-5
    h = hessian(x)
    g = gradient(x)

    dx = (-matmul((h + u * np.identity(50)).I, g)).A1
    while value(x) <= value(x + dx):
        u *= 1.2
        dx = (-matmul(h + u * np.identity(50), g)).A1
    print(dx)
    return dx


def steepest_descent_l1(x):
    s = gradient(x)
    ind = -1
    m = 0

    for i in range(len(s)):
        if abs(s[i]) > m:
            ind = i
            m = abs(s[i])
    # ind = np.argmax(s)
    dx = np.zeros(len(s))
    dx[ind] = -s[ind]
    # print(dx)
    # np.argmax(s)

    # print("::", getAlpha(x, s, c1=0.001) * s)
    return backtrack(x, dx, alpha=0.5, beta=0.9) * dx


def steepest_descent_l2(x):
    return backtrackLine(x)


def conjugateGradient(x):
    g = gradient(x)

    r = b - np.dot(A, x)  #r(k+1)=b-A*x(k+1)
    beta = np.linalg.norm(r)**2 / np.linalg.norm(r1)**2
    d = r + beta * d

    return getAlpha(x, d, c1=0.5) * d


def train(method, x, maxiter=100000, err=1e-6):
    dx = method(x)

    i = 0
    ans = []
    while value(x) - 0 >= err and i <= maxiter:
        # if i % 100 == 0:
        print(i, value(x))
        ans.append(value(x))
        i += 1

        x = x + dx

        dx = method(x)  # next step

    return ans, x


if __name__ == "__main__":
    starter = np.array([-1.2, 1.0] * 25)

    methods = []
    # methods.append(exactLine)
    # methods.append(backtrackLine)
    # methods.append(dampedNewton)
    # methods.append(gaussNewton)   

    # methods.append(levenbergMarquardt) # ??
    methods.append(steepest_descent_l1) # ??
    # methods.append(steepest_descent_l2) 
    # methods.append(conjugateGradient) # ??

    for method in methods:
        starter = np.array([-1.2, 1.0] * 25)
        err, ans = train(method, starter)
        report(method, err)