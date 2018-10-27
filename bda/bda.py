import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

x = np.array([-0.86, -0.30, -0.05, 0.73])
n = np.array([5, 5, 5, 5])
y = np.array([0, 1, 3, 5])

ngrid = 100
A = np.linspace(-4,4, ngrid)
B = np.linspace(-10,30,ngrid)

# elements of join prior
ab_mean = np.array([0,10])
ab_cov = np.array([4,10], [10,100])

# create empty grid
lp_grid = np.ones((len(A), len(B)))
for i in range (len(lp_grid)): # for each row, which 100 column
    for j in range: # for each column on that row
        a = A[j]
        b = B[i]
        t = a+b*x
        et = np.exp(t)