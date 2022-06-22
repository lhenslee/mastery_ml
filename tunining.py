"""
Chapter 16: Improve with Tuning
- Grid Search Parameter tuning
- Random Search Parameter Tuning: random for fixed number of iter
"""
import numpy as np
from scipy.stats import uniform
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from get_data import X, y

print('Grid Search Parameter Tuning')
alphas = np.array([1, .1, .01, .001, .0001, 0])
param_grid = dict(alpha=alphas)
model = Ridge()
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(X, y)
print(grid.best_score_)
print(grid.best_estimator_.alpha)

print('Random Search Parameter Tuning')
param_grid = {'alpha': uniform()}
model = Ridge()
rsearch = RandomizedSearchCV(
    estimator=model, param_distributions=param_grid, n_iter=100, random_state=7)
rsearch.fit(X, y)
print(rsearch.best_score_)
print(rsearch.best_estimator_.alpha)
