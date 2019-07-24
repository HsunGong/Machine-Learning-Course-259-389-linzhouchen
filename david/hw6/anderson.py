import numpy as np
from math import *

from numpy.linalg import inv, norm, solve


# % [xnew] = anderson(X, G)
# %
# % Apply Anderson acceleration to the fixed point iteration
# %   xnew = g(x)
# % given a history X = [x(k-m), ..., x(k)] and
# % corresponding evaluations [G(x(k-m)), ..., G(x(k))].
# % Returns the new point xnew = x(k+1).
# %
def anderson(X, G):
    #   % Ref: Homer F. Walker and Peng Ni,
    #   % "Anderson Acceleration for Fixed-Point Iterations"
    #   % https://doi.org/10.1137/10078356X
    #   %
    #   % NB: This all could be done a bit more efficiently by an update
    #   %     scheme.  In the interest of clarity, though, we're not
    #   %     going to bother.
    print(G.shape)
    F = G - X
    DF = F[:, 1:] - F[:, 0: - 1]
    DG = G[:, 1:] - G[:, 0:-1]
    fk = F[:, -1]

    # xnew = G[:, -1] - DG.dot(inv(DF).dot(fk))
    xnew = G[:, -1] - DG.dot(solve(DF, fk))

    return xnew

import os, sys
sys.path.append('../')
from hw6.hals import *

from numpy import array

AH0 = [H0]
GH0 = []
def updateAH(R, W, H):
    # print('update-H ', W, H)
    M = (W.T).dot(R)
    N = (W.T).dot(W)
    u = []
    for row in range(H.shape[0]):
        ui = []
        for col in range(H.shape[1]):
            ui.append(max(-H[row, col], M[row, col] / N[row, row]))
        u.append(ui)
    u = np.array(u)

    GH = H + u
    GH0.append(GH)

    AH = [] # certain row for all times
    for row in range(GH.shape[0]):
        AH.append(anderson(array(AH0)[:, row, :].T, array(GH0)[:, row, :].T).T)
        # print(anderson(array(AW0)[:, row, :].T, np.array(GW0)[:, row, :].T).T.shape)
    AH0.append(AH)

    return np.array(AH)


AW0 = [W0]
GW0 = []
def updateAW(R, W, H):
    M = R.dot(H.T)
    N = H.dot(H.T)
    u = []
    for row in range(W.shape[0]):
        ui = []
        for col in range(W.shape[1]):
            ui.append(max(-W[row, col], M[row, col] / N[col, col]))
        u.append(ui)
    u = np.array(u)

    GW = W + u
    GW0.append(GW)

    AW = [] # certain row for all times
    for row in range(GW.shape[0]):
        AW.append(anderson(array(AW0)[:, row, :].T, array(GW0)[:, row, :].T).T)
        # print(anderson(array(AW0)[:, row, :].T, np.array(GW0)[:, row, :].T).T.shape)
    AW0.append(AW)

    return np.array(AW)


r = [norm(R0, ord='fro')]
max_iter = 100  # or error
cnt = 0

while cnt < max_iter:
    cnt += 1

    W0 = updateAW(R0, W0, H0)
    R0 = A - W0.dot(H0)

    H0 = updateAH(R0, W0, H0)
    R0 = A - W0.dot(H0)

    r.append(norm(R0, ord='fro'))

import matplotlib.pyplot as plt
plt.semilogy([rr - r[-1] for rr in r])
plt.xlabel("step")
plt.ylabel("error")
plt.savefig('anderson.png')
plt.show()

[
    [1,2,3,4],
    [2,3,4,5],
    [3,4,5,6]
]

[
    [0,2,0,4],
    [2,0,4,0],
    [2,4,1,6]]
