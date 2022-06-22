"""
Chapter 12: Spot-Check Regression Algorithms
- Linear Machine Learning Algorithms
  - Linear Regression
    - Assumes inputs have Gaussian distribution
    - Low correlation (problem is collinearity)
  - Ridge Regression
    - Minimize complexity of model
      - Sum squared value of coefficient values (L2-norm)
  - Least Absolute Shrinkage and Selection Operator
    - Sum absolute value of coefficient values (L1-norm)
  - ElasticNet Regression
    - L2-norm and L1-norm to minimize complexity
- Nonlinear Machine Learning Algorithms
  - K-Nearest Neighbors
    - Minkowski distance (Euclidean mixed with Mahatten)
    - Euclidean distence: all inputs have same scale
    - Mahatten: when scales differ
  - Classification and Regression Trees
    - Select best points to split data
  - Support Vector Machines (SVM)
"""
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from get_data import boston_df
X = boston_df.values[:, :13]
y = boston_df.values[:, 13]

print('Linear Regression')
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = LinearRegression()
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
print(results.mean())

print('Ridge Regression')
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = Ridge()
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
print(results.mean())

print('LASSO Regression')
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = Lasso()
results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
print(results.mean())

print('ElasticNet Regression')
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = ElasticNet()
results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
print(results.mean())


print('K-Nearest Neighbors')
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = KNeighborsRegressor()
results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
print(results.mean())

print('Classification and Regression Trees')
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = DecisionTreeRegressor()
results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
print(results.mean())

print('Support Vector Machine')
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = SVR()
results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
print(results.mean())
