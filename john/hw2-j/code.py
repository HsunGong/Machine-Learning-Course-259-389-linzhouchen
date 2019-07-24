import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import style
from math import *
import random

#%% Q 16
import numpy as np
from numpy.linalg import svd, norm, eig

A = np.matrix([[1, 2], \
                [-1, 2], \
                [1, -2], \
                [-1, -2], ])

U, sigma, VT = svd(A, full_matrices=True)

B = A.T * A
x = np.matrix([1, 1]).T

# 1
step = 0
err = 10.0
while err > 0 and step < 3:
    v1 = np.dot(B, x) / norm(np.dot(B, x))
    # print(v1)
    err = norm(x - v1)
    x = v1
    step += 1

print(VT)
print(x)

# 2
print(B)
ei, ev = eig(B)
print(ei, ev)

# print(U, Sigma, VT)
print(U.shape[1], VT.shape[0])
Sigma = np.zeros((U.shape[1], VT.shape[0]))
Sigma[0][0] = 4
Sigma[1][1] = 2
print(np.dot(U, np.dot(Sigma, VT)))
# print(Sigma)

#%% Q 18
import numpy as np
from numpy.linalg import svd, norm, eig
import random
import matplotlib.pyplot as plt

pppp = 1
def plot(x, y, iter, pppp):
    pp = plt.subplot(2, 3, pppp)

    pp.set_title('step: ' + str(iter))
    for i in range(len(x)):
        pp.plot(x[i:i + 2], y[i:i + 2], color='r')
        pp.scatter(x[i], y[i], color='b')


N = np.linspace(5, 25, num=5, dtype=int)
N = [5]
max_step = range(250)
for n in N:
    plt.figure(figsize=(20, 20))
    x = np.random.normal(0, 1, (n, 1))
    y = np.random.normal(0, 1, (n, 1))
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    x /= np.linalg.norm(x)
    y /= np.linalg.norm(y)
    
    size = range(n)

    pppp = 1
    for iter in max_step:
        if iter % 50 == 0:
            plot(x, y, iter, pppp)
            print(pppp, iter)
            pppp = pppp + 1

        x = np.array([0.5*(x[i] + x[i+1]) for i in size])
        y = np.array([0.5*(y[i] + y[i+1]) for i in size])
        x /= np.linalg.norm(x)
        y /= np.linalg.norm(y)
        x = np.append(x, x[0])
        y = np.append(y, y[0])


    plot(x, y, iter, pppp)
    plt.savefig('18-2.png')
    plt.show()

#%% Q 18-2

import numpy as np
from numpy.linalg import svd

A = np.diag([0.5]*5) + np.diag([0.5]*4, k=1) + np.diag([0.5], k=4).T
A = np.matrix(A)
print('A', A.I)

u, s, vt = svd(A)
print(s)
print(vt.T[1])

#%% Q 28
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import *


def get_approx_matrix(u, sigma, v, rank):  #rank
    m = len(u)
    n = len(v)
    a = np.zeros((m, n))
    k = 0
    while k < rank:
        uk = u[:, k].reshape(m, 1)
        vk = v[k].reshape(1, n)
        a += sigma[k] * np.dot(uk, vk)
        k += 1

    a[a < 0] = 0
    a[a > 255] = 255
    return a.astype("uint8")


def get_svd_image(file_path):
    name, suffix = file_path.split(".")
    img = Image.open(file_path, 'r')
    a = np.array(img)

    u0, sigma0, v0 = np.linalg.svd(a[:, :, 0])
    u1, sigma1, v1 = np.linalg.svd(a[:, :, 1])
    u2, sigma2, v2 = np.linalg.svd(a[:, :, 2])
    print("origin pic-> red rank:", len(sigma0), " green_rank:", len(sigma1),
          " blue_rank:", len(sigma2))

    for rank in [1, 2, 4, 16]:
        red_matrix = get_approx_matrix(u0, sigma0, v0, rank)
        green_matrix = get_approx_matrix(u1, sigma1, v1, rank)
        blue_matrix = get_approx_matrix(u2, sigma2, v2, rank)

        ans = sqrt(np.linalg.norm(sigma0[:rank])) + \
                sqrt(np.linalg.norm(sigma1[:rank])) + \
                    sqrt(np.linalg.norm(sigma2[:rank]))
        print('rank: ' + str(rank) + ' F: ' + str(ans / 3))

        I = np.stack((red_matrix, green_matrix, blue_matrix), 2)
        Image.fromarray(I).save(str(rank) + file_path)


import os
os.chdir('./hw2-j') if os.listdir().count('hw2-j') != 0 else None

name = "pic"
suf = '.jfif'
get_svd_image(name + suf)
img = mpimg.imread(name + suf)
plt.imshow(img)
plt.savefig('pic.png')

plt.figure(figsize=(20, 20))
x = 0
for i in [1, 2, 4, 16]:
    ax = plt.subplot(2, 2, x + 1)
    img = mpimg.imread(str(i) + name + suf)
    ax.imshow(img)
    ax.set_title("rank:" + str(i * 5))
    ax.set_axis_off()
    x = x + 1
plt.savefig('28.png')
plt.show()

#%% Q 32
import os
os.chdir('./hw2-j') if os.listdir().count('hw2-j') != 0 else None

import matplotlib.pyplot as plt

import numpy as np


def cmdscale(D):
    """                                                                                       
    Classical multidimensional scaling (MDS)                                                  
                                                                                               
    Parameters                                                                                
    ----------                                                                                
    D : (n, n) array                                                                          
        Symmetric distance matrix.                                                            
                                                                                               
    Returns                                                                                   
    -------                                                                                   
    Y : (n, p) array                                                                          
        Configuration matrix. Each column represents a dimension. Only the                    
        p dimensions corresponding to positive eigenvalues of B are returned.                 
        Note that each dimension is only determined up to an overall sign,                    
        corresponding to a reflection.                                                        
                                                                                               
    e : (n,) array                                                                            
        Eigenvalues of B.                                                                     
                                                                                               
    """
    # Number of points
    n = len(D)

    # Centering matrix
    H = np.eye(n) - np.ones((n, n)) / n

    # YY^T
    B = -H.dot(D**2).dot(H) / 2

    # Diagonalize
    evals, evecs = np.linalg.eigh(B)

    # Sort by eigenvalue in descending order
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Compute the coordinates using positive-eigenvalued components only
    w, = np.where(evals > 0)
    L = np.diag(np.sqrt(evals[w]))
    V = evecs[:, w]
    Y = V.dot(L)

    return Y, evals


dic = {}
with open('data.txt', 'r') as f:
    flag = True
    for line in f.readlines():
        if not line.split():
            flag = False
            continue
        data = line.strip('\n').split(' ')
        dic[data[0]] = (data[1:] if flag else dic[data[0]] + data[1:])

    for k, vlist in dic.items():
        newv = []
        for v in vlist:
            try:
                newv.append(int(v))
            except ValueError as e:
                newv.append(0)
        dic[k] = newv

dis = np.array([v for v in dic.values()])
cities = [k for k in dic.keys()]
size = len(cities)

Y, eigvals = cmdscale(dis)

print(Y.shape)

# for d in dis:
#     print(d)

# for eig in eigvals:
#     print([eig, eig/max(abs(eigvals))])

plt.scatter([y[0] for y in Y], [y[1] for y in Y], color='b')
for i in range(size):
    plt.annotate(cities[i], ([y[0] for y in Y][i], [y[1] for y in Y][i]))
plt.xlabel('Miles')
plt.ylabel('Miles')
plt.savefig('32.png')
plt.show()

#%%
