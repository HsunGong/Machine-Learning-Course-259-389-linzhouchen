# LADMPSAP
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
    plt.ylabel('$log-y$')

    # f* = 0
    plt.plot((range(len(err))), np.log(err))

    plt.savefig(name)
    plt.show()
    return


import numpy.random as rand
np.random.seed(0)

value = lambda U, V, A : 0.5 * norm(U @ V - A, 'fro')**2

# Omega = 0.1*m*n
# P(A) = P(D) in Omega -> P_Omega
Omega = np.random.randint(0,2,size=(200, 300))
Omega = (Omega == 1)


def P(x):
    x[Omega] = 0
    return x

m = 200
n = 300

U0 = rand.uniform(0, 1, size=(m,5))
V0 = rand.uniform(0, 1, size=(5,n)) # V>T

A0 = rand.uniform(0, 1, size=(m,n)) # V>T

# Question is equal to min rank(A) s.t. P(D-A) = 0
# D is the final value, and we'd like to optim it into A which s.t. P(D-A) = 0
# K= 5
D = U0 @ V0
from numpy.linalg import matrix_rank as rank
print(rank(D))

from numpy.linalg import pinv
def Coordinate(U, V, A, maxiter=100):
    err = 1e-5
    ans = [value(U, V, A)]  # 0
    print(ans)
    i = 0
    while i <= maxiter:

        i += 1

        U = A @ pinv(V)
        V = pinv(U) @ A
        A = U @ V + P(D - U @ V)

        print(i, value(U, V, A))
        ans.append(value(U,V,A))

        if norm((D-A), 'fro') / norm(P(D), 'fro') <= err: # and abs(1-norm(P(D-A), 'fro') / norm(P(D- Uold @ Vold), 'fro')) <= err/2:
            break

    return ans, [U,V,A]



err, ans = Coordinate(U0, V0, A0)
print(norm(ans[2] - D, 2))
report("Coordinate", err)
