"""
Chapter 9: Evaluate Performance
- Train, test, retrain on all data
- Train/test split 67%/33%
- K-fold cross validation
  - Splits dataset num of fold times and summarizes over multiple tests
  - Common values are 3, 5, and 10
  - More reliable on unseen data
- Leave One Out Cross Validation
  - More variance
  - Best but slow
- Repeated random test-train splits
- When to use what
  - k-fold is the gold standard with k set to 3, 5, 10
  - train/test split is good for speed
  - 10 fold is the best
"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, LeaveOneOut, ShuffleSplit, cross_val_score, train_test_split
from get_data import X, y

seed = 7
num_folds = 10

print('67 - 33 train - test')
test_size = .33
X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=test_size, random_state=seed)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)
result = model.score(X_test, Y_test)
print('Accuracy:', result*100)

print('K-fold Cross Validation')
kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
model = LogisticRegression(max_iter=1000)
results = cross_val_score(model, X, y, cv=kfold)
print('Accuracy:', results.mean()*100, results.std()*100)

print('Leave One Out')
loocv = LeaveOneOut()
model = LogisticRegression(max_iter=1000)
results = cross_val_score(model, X, y, cv=loocv)
print('Accuracy:', results.mean()*100, results.std()*100)

print('Repeated Random Test-Train Splits')
n_splits = 10
kfold = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
model = LogisticRegression(max_iter=1000)
results = cross_val_score(model, X, y, cv=kfold)
print('Accuracy:', results.mean()*100, results.std()*100)
