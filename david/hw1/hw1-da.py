import matplotlib.pyplot as plt

from scipy.stats import linregress

def _1():
    A = np.matrix([[6, 3], [1, 3]])
    E = np.matrix([[1, 0], [0, 0]])

    p = range(-30, 50) # 0 - 9
    x = [exp(i) for i in p]
    # print(x)
    y = [np.linalg.norm((A + h*E).I - (A.I - h*A.I*E*A.I)) for h in x]
    

    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    print(slope, intercept)

    plt.plot([log(h) for h in x], [log(h) for h in y])
    plt.xlabel("log(h)")
    plt.ylabel("log(y)")
    plt.title("A simple plot")
    plt.savefig("test.png", dpi=120)
    plt.show()


if __name__ == "__main__":
    try:
        print("Problem 1:")
        _1()
    except KeyboardInterrupt as e:
        pass
