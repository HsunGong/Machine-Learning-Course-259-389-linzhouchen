#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import style
from math import *
import random

#%%

def _1():
    sample_num = 100
    x = np.linspace(-4, 4, sample_num)

    step_size = 2e5
    learn_rate = 1e-4
    batch_size = 20

    cnt = 0
    step = []
    c = np.array([1.0, 0, 0])
    Y = []
    y0 = 0.5 / sample_num * np.linalg.norm([np.dot(c.T, [1, xi**2, xi**4]) - cos(xi) for xi in x], 2)
    while cnt < step_size:
        # fresh batch with w
        index = random.sample(range(sample_num), batch_size) # select some 20 batch
        batch_x = x[index]
        
        # gradient
        # xi = batch_x[0] 
        # print(np.dot(c.T, [1, xi**2, xi**4]))
        batch_dy = np.array([np.array([1, xi**2, xi**4]) * (np.dot(c.T, [1, xi**2, xi**4]) - cos(xi)) for xi in batch_x]) # gred: c_1, c_2, c_3, 0

        # descent
        descent_all = np.array([0.0]*3)
        for dy in batch_dy:
            descent_all += dy
        descent_all /= batch_size
        c -= learn_rate * descent_all

        Y.append(0.5 / sample_num * np.linalg.norm([np.dot(c.T, [1, xi**2, xi**4]) - cos(xi) for xi in x], 2)**2)
        
        print("step: ", cnt, "the new func:", Y[-1] - y0)
        cnt += 1
        step.append(cnt)

    print(step)

    plt.plot([log(s) for s in step],[y - y0 for y in Y])
    plt.xlabel("step")
    plt.ylabel("error")
    plt.title("Res")
    plt.savefig("hw.png", dpi=360)
    plt.show()

if __name__ == '__main__':
    _1()


