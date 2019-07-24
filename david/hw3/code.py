import matplotlib.pyplot as plt

import numpy as np
from math import *
from numpy.random import normal, multivariate_normal, uniform
from numpy.linalg import norm
from sklearn.preprocessing import normalize

from scipy.stats import linregress

def _1():
    r = 0.05
    X = np.arange(0, 1+r, r)
    K = range(1, 11)

    F = [np.array([0]*len(X))]
    for k in K:
        gt = np.array([1/k * mul(1, k - 1, x) * x for x in X]) # start at 0, but default is 1(if smaller, ans = 1)
        
        f = gt + F[k-1]

        F.append(f)

    for k in K:
        plt.plot(X, F[k], color='C'+str(k-1), label=str(k))
    
    plt.xlabel("x")
    plt.ylabel("f")
    plt.title("Res")
    plt.savefig("hw3.png", dpi=360)
    plt.show()
#%%
def mul(start, end, x) :
    ans = 1.0
    for cnt in range(start, end+1):
        # print(cnt)
        # print(ans)
        ans *= (1 - 1/cnt * x**2)
    # print(start, end, ans)
    return ans

#%%
if __name__ == "__main__":
    try:
        print("Problem 1:")
        _1()
    except KeyboardInterrupt as e:
        pass
