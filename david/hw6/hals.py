A = [[0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1], [1, 0, 1, 0, 0],
     [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 1, 1, 0], [0, 1, 1, 0, 0],
     [0, 0, 1, 1, 1], [0, 1, 1, 0, 0]]

import numpy as np
from math import *
from numpy.linalg import norm


def svd_nmf_init(U, S, V, k=-1):

    sig = np.diag(S)
    if k == -1:
        k = len(sig)

    W = np.zeros((U.shape[0], k))
    H = np.zeros((k, V.shape[1]))

    for j in range(k):
        x = U[:, j]
        xp = np.max(x, 0)
        xn = -np.min(x, 0)
        y = V[:, j]
        yp = np.max(y, 0)
        yn = -np.min(y, 0)
        norm_xp = norm(xp)
        norm_yp = norm(yp)
        norms_p = norm_xp * norm_yp
        norm_xn = norm(xn)
        norm_yn = norm(yn)
        norms_n = norm_xn * norm_yn
        if norms_p > norms_n:
            scale = sqrt(sig[j, j] * norms_p)
            W[:, j] = scale * xp / norm_xp
            H[j, :] = scale * yp.T / norm_yp
        else:
            scale = sqrt(sig[j, j] * norms_p)
            W[:, j] = scale * xp / norm_xp
            H[j, :] = scale * yp.T / norm_yp

    return W, H


# SVD-based initial factorization
from numpy.linalg import svd

k = 3
U, S, VT = svd(A)
V = VT.T
W0, H0 = svd_nmf_init(U, S, V, k)

W0 = [[0.2408, 0.3451, 0], [0.1329, 0.3703, 0.5066], [0.1329, 0.3703, 0.5066],
      [0.6629, 0, 0], [0.2191, 0.1051, 0], [0.1724, 0, 0.2533],
      [0.9037, 0.3062, 0], [0.6162, 0, 0.2533], [0.8175, 0.5714, 0.5066],
      [0.6162, 0, 0.2533]]
H0 = [
    [0.6255, 0.4921, 1.2668, 0.6873, 0.3795],
    [0.1846, 0, 0, 0.6061, 0.6504],
    [0, 0.4387, 0.0000, 0, 0.8774],
]
W0 = np.array(W0)
H0 = np.array(H0)
R0 = A - W0.dot(H0)

# HALS-RRI iteration (you do!)


def updateW(R, W, H):
    M = R.dot(H.T)
    N = H.dot(H.T)
    u = []
    for row in range(W.shape[0]):
        ui = []
        for col in range(W.shape[1]):
            ui.append(max(-W[row, col], M[row, col] / N[col, col]))
        u.append(ui)
    u = np.array(u)

    W = W + u
    return W


def updateH(R, W, H):
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

    # print('old ', H)
    H = H + u
    # print('new ', H)
    return H


import matplotlib.pyplot as plt
if __name__ == "__main__":

    r = [norm(R0, ord='fro')]
    max_iter = 100  # or error
    cnt = 0
    while cnt < max_iter:
        cnt += 1

        W0 = updateW(R0, W0, H0)
        R0 = A - W0.dot(H0)
        H0 = updateH(R0, W0, H0)
        R0 = A - W0.dot(H0)
        print(R0)
        r.append(norm(R0, ord='fro'))

    # plot
    plt.semilogy([rr - r[-1] for rr in r])
    plt.xlabel("step")
    plt.ylabel("error")
    plt.savefig('hals.png')
    plt.show()



