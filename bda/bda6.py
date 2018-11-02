import numpy as np
import pystan

data = {'x': [-0.86, -0.30, -0.05, 0.73],
        'y': [0, 1, 3, 5],
        'n': [5, 5, 5, 5],
        'mean': [0, 10],
        'cov': [[4, 10],[10, 100]]}
bs = pystan.StanModel(file='model.stan')
fit = bs.sampling(data=data, iter=1000, chains=4)
fit.plot()
# schools_dat = {'J': 8,
#                'y': [28,  8, -3,  7, -1,  1, 18, 12],
#                'sigma': [15, 10, 16, 11,  9, 11, 10, 18]}