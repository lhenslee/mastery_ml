"""
Chapter 15: Improve Performance with Ensembles
- Bagging: Build same type models from different subsamples
  - Bagged decision trees
  - Random Forest: extension of bagged, less correlation
  - Extra Trees
- Boosting: Build same type models from fixing errors
  - AdaBoost: Weight on how easy to classify
  - Stochastic Gradient: One of the best
- Voting: Build different type models and combine predictions
"""
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from get_data import X, y

print('Bagged decision trees')
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(
    base_estimator=cart, n_estimators=num_trees, random_state=7)
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())

print('Random Forest')
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
max_features = 3
model = RandomForestClassifier(
    n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())

print('Extra Trees')
max_features = 7
model = ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())

print('ADA Boost')
model = AdaBoostClassifier(n_estimators=num_trees, random_state=7)
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())

print('Stochastic Gradient Boosting')
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=7)
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())

print('Voting Ensemble')
estimators = [
    ('logistic', LogisticRegression(max_iter=1000)),
    ('cart', DecisionTreeClassifier()),
    ('svm', SVC())
]
ensemble = VotingClassifier(estimators)
results = cross_val_score(ensemble, X, y, cv=kfold)
print(results.mean())
