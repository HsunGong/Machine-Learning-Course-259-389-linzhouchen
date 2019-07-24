# LADMPSAP
import matplotlib.pyplot as plt

import numpy as np
from numpy import matmul
from numpy.linalg import norm


import sys, os.path
def report(name, err):
    name = os.path.splitext(sys.argv[0])[0] + " model-" + name

    plt.figure()
    plt.title(name)
    plt.colorbar
    plt.xlabel('$x$')
    plt.ylabel('$y$')

    # f* = 0
    plt.plot(range(len(err)), (err))

    plt.savefig(name)
    plt.show()
    return

m = 200
n = 300


def solve_l2(w, t):
    nw = norm(w, 1)
    if nw > t:
        return (nw - t) * w / nw
    else:
        return np.zeros(w.size())


def solve_l1l2(W, t):
    # % min lambda |x|_2 + |x-w|_2^2
    E = W
    for i in range(n):
        E[:, i] = solve_l2(W[:, i], t)


from scipy.sparse.linalg import svds


def ladmp_lrr(D, t=0.1, rho=1.9, DEBUG=False):
    # t means lambda
    '''
    min |Z|_*+lambda*|E|_2,1
    s.t., D = DZ+E
    
    D: m*n
    Z: n*n
    E: m*n
    '''
    #M=Z_k+X'*(X-X*Z_k-E_{k+1}+Y/mu_k)/eta.

    normfD = norm(D, 'fro')
    norm2D = norm(D, 2)
    tol1 = 1e-4  # %threshold for the error in constraint
    tol2 = 1e-5  # %threshold for the change in the solutions
    maxiter = 1000
    max_mu = 1e10

    mu = min(m, n) * tol2
    eta = norm2D**2 * 1.02
    # %eta needs to be larger than ||D||_2^2, but need not be too large.

    E = np.zeros((m, n))
    Y = np.zeros((m, n))
    Z = np.zeros((n, n))
    DZ =np.zeros((m, n))  # D * Z

    sv = 5
    svp = sv

    convergenced = 0
    iter = 0

    while iter < maxiter:
        iter += 1

        Ek = E
        Zk = Z

        E = solve_l1l2(D - DZ +  Y / mu, t / mu) # todo

        # propack
        # print(D.shape, E.shape)
        # print(D  - E)
        M = Z + D.T @ (D - DZ - E + Y / mu) / eta
        # opt.tol = tol2;%precision for computing the partial SVD
        # opt.p0 = ones(n,1);
        U, S, Vt = svds(M, n, n, ncv=sv, tol=tol2, which='LM', v0 = np.ones(n))
        V = Vt.T
        # S = np.diag(S)
        svp = len(S[S > 1/(mu*eta)])
        if svp < sv:
            sv = min(svp + 1, n)
        else:
            sv = min(svp + round(0.005*n, 0), n)
        
        if svp >= 1:
            S = S[0:svp] - 1/(mu*eta)
        else:
            svp = 1
            S = 0

        AU = U[:, 0:svp]
        As = S
        AV = V[:, 0:svp]

        Z = AU @ np.diag(As) @ AV.T

        diffZ = norm(Zk - Z, 'fro')

        relChgZ = diffZ / normfD
        relChgE = norm(E - Ek, 'fro') / normfD
        relChg = max(relChgZ, relChgE)

        DZ = D @ Z
        dY = D - DZ - E
        recErr = norm(dY, 'fro') / normfD

        convergenced = recErr < tol1 and relChg < tol2


        if convergenced:
            break
        else:
            Y = Y + mu @ dY
            if mu * relChg < tol2:
                mu = min(max_mu, mu*rho)


import numpy.random as rand
np.random.seed(0)

D = rand.uniform(0, 1, (m, n))

err, ans = ladmp_lrr(D)

report("Ladmpsap", err)