import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from random import randint

x = np.array([-0.86, -0.30, -0.05, 0.73])
n = np.array([5,5,5,5])
y = np.array([0,1,3,5])
ngrid = 100
A = np.linspace(-4,4,ngrid)
B = np.linspace(-10,30,ngrid)
ab_mean = np.array([0,10])
ab_cov = np.array([[4,10], [10, 100]])

def get_multivar_target_dist_logpdf(A, B, ab_mean, ab_cov):
    lp_grid = np.ones((len(A), len(B)))
    for i in range(len(lp_grid)):
        for j in range(len(lp_grid[i])):
            a = A[j]
            b = B[i]
            t = a+b*x
            et = np.exp(t)
            z = et/(1. + et)
            lp = np.sum(y*np.log(z) + (n-y) * np.log(1.0-z), axis=-1)
            ab_prior = np.log(multivariate_normal.pdf([a,b], ab_mean, ab_cov))
            lp = lp + ab_prior
            lp_grid[i,j] = lp

    ## We taking log_densities here so no need to invert
    # lp_grid = np.exp(lp_grid - np.amax(lp_grid))
    lp_grid = lp_grid/np.sum(lp_grid)

    return lp_grid
    
target_dist = get_multivar_target_dist_logpdf(A, B, ab_mean, ab_cov)

# Metropolis algorithm
start_x_idx, start_y_idx = 30, 30 # totally arbitrary here
start = target_dist[start_x_idx][start_y_idx]
N = 400
samples = np.zeros((len(A), len(B)))
samples_grid_points = [[A[start_x_idx], B[start_y_idx]]]
samples[start_x_idx][start_y_idx] = start
for i in range(N):
    # jumping dist is a Bivariate normal dist scaled to 0.5 the size
    # centered at current iteration
    jumping_mean = [A[start_x_idx], B[start_y_idx]]
    jumping_cov = 0.5 ** 2 * ab_cov
    jumping_dist = get_multivar_target_dist_logpdf(A, B, jumping_mean, jumping_cov)

    # sample from jumping dist
    sample_x = randint(0,99)
    sample_y = randint(0,99)
    jumping_dist_sample = jumping_dist[sample_x][sample_y]

    # calc ratio and compare with acceptance threshold
    rand_threshold = np.random.rand()
    r = jumping_dist_sample / start
    if (r > rand_threshold):
        new_start = target_dist[sample_x][sample_y]
        start_x_idx = sample_x
        start_y_idx = sample_y
        start = new_start
        samples[start_x_idx][start_y_idx] = start
        samples_grid_points.append([A[start_x_idx], B[start_y_idx]])

points = np.array(samples_grid_points)
grid_x = points[:,0]
grid_y = points[:,1]
fig, ax = plt.subplots()
ax.scatter(
    grid_x,
    grid_y
)

plt.show()
foo = 5

