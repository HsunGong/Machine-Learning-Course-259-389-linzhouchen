#%% init data
from sympy import *
from fractions import Fraction

h = symbols('h', real=True)

f = (1 / (5 + h) - 1 / 5 + h / 25)**2 + (-1 / (5 + h) + 1 / 5 - h / 25)**2 + (
    -1 / (3 * (5 + h)) + Fraction('1/15') - h * Fraction('1/75'))**2 + (
        (6 + h) / (3 * (5 + h)) - 2 / 5 + h * Fraction('1/75'))**2

# print(f)

#%%
f = simplify(f)

print(f)

h = symbols('h', real=True)

# A = Matrix([[6, 3], [1, 3]])
# E = Matrix([[1, 0], [0, 0]])
# B = A**-1

# f = (A + h * E)**(-1) - B + h * B * E * B
# f = simplify(f)
# print(f)

# print(f.norm(2))

#%%

f = (h**2 /
     (25 *
      (h + 5)))**2 + (h**2 /
                      (25 * h + 125))**2 + (h**2 /
                                            (75 * h + 375))**2 + (h**2 /
                                                                  (75 *
                                                                   (h + 5)))**2

f = simplify(f**(1 / 2))

print(f)
