"""
Chapter 11: Spot-Check Classification
- Linear
  - Logistic Regression
    - Assumes Gaussian distribution
  - Linear Discriminant Analysis (LDA)
    - Binary and multi-class
    - Also assume Gaussian
- Nonlinear
  - k-Nearest Neighbors (KNN)
    - Uses distance like title
  - Naive Bayes
    - Calculates probability of each class
  - Classification and Regression Trees (CART/decision trees)
    - Greedily chooses each attribute
  - Support Vector Machines (SVM)
    - Seek a line between classes
"""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from get_data import X, y

print('Logistic regression')
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = LogisticRegression(max_iter=1000)
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())

print('Linear Discriminant Analysis')
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = LinearDiscriminantAnalysis()
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())

print('k-Nearest Neighbors')
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = KNeighborsClassifier()
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())

print('Naive Bayes')
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = GaussianNB()
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())

print('Classification and Regression Trees')
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = DecisionTreeClassifier()
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())

print('Support Vector Machines')
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = SVC()
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())
