import numpy as np
from math import *
from numpy.random import normal, multivariate_normal, uniform
from numpy.linalg import norm
from sklearn.preprocessing import normalize


num = 20
d = 900


def _38():
    r = 30
    k = [100, 50, 10, 5, 4, 3, 2, 1]

    m = [0] * d
    cov = np.diag([1] * d)
    dots = []
    for i in range(num):
        dot = normal(0, 1, d)
        scale = 30 / norm(dot)
        dots.append(dot*scale)

    distance = []
    for i in range(num):
        for j in range(i + 1, num):
            distance.append(norm(dots[i] - dots[j]))

    for dim in k:
        old_dis = [i * math.sqrt(dim) for i in distance]
        new_dis = getDis(dim, dots)

        diff = []
        for x in range(len(new_dis)):
            diff.append(abs(old_dis[x] - new_dis[x]))

        ans = max(diff) / math.sqrt(dim)
        print(round(ans, 0))


def getDis(dim, old_dot):
    # dim = sample num, d-dimensional
    orth_base = multivariate_normal([0]*d, np.diag([1]*d), dim)

    new_dot = []
    for dot in old_dot:
        new_dot.append(
            np.array([np.dot(orth_base[i], dot) for i in range(dim)]))
    # print((norm(new_dot[-1]) - norm(dot) * math.sqrt(dim) )/ norm(new_dot[-1]) )

    new_dis = []
    for i in range(num):
        for j in range(i + 1, num):
            new_dis.append(norm(new_dot[i] - new_dot[j]))

    return new_dis


def _40():
    dim = 100
    dr = 1000
    # way 1
    s = normal(0, 1, (dr, dr))

    orth_base = normal(0, 1, (dim, dr))

    new_dot = []
    for dot in s:
        new_dot.append(
            np.array([np.dot(orth_base[i], dot) for i in range(dim)]))

    deg = []
    for i in range(num):
        for j in range(i + 1, num):
            deg.append(cos(new_dot[i], new_dot[j]))

    print(math.acos(max(deg)))

    # way 2
    new_dot = normal(0, 1, (dim, dim))

    deg = []
    for i in range(num):
        for j in range(i + 1, num):
            deg.append(cos(new_dot[i], new_dot[j]))

    print(math.acos(max(deg)))


def cos(a, b):
    return norm(a - b) / (norm(a) + norm(b))

if __name__ == "__main__":
    try:
        print("Problem 38:", end='\n')
        _38()
        print("\nProblem 40:", end='\n')
        _40()
    except KeyboardInterrupt as e:
        pass
