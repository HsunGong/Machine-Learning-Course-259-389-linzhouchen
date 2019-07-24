import numpy as np

a = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]])
b = np.array([[0, 2, 0, 4], [2, 0, 4, 0], [2, 4, 1, 6]])
from numpy.linalg import inv, norm, solve

def anderson(X, G):
    F = G - X
    DF = F[:, 1:] - F[:, 0: - 1]
    DG = G[:, 1:] - G[:, 0:-1]
    fk = F[:, -1]

    xnew = G[:, -1] - DG.dot(solve(DF, fk))
    print(G[:,-1])
    print(inv(DF).dot(fk))
    print(DG.dot(inv(DF).dot(fk)))
    
    return xnew

print('xnew', anderson(a, b))