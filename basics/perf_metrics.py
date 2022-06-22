"""
Chapter 10: Performance Metrics
- Classification Accuracy
  - Only suitable when there are an equal classes
  - Ratio of correct to total
- Logarithmic Loss
  - Predictions rewarded/punished according to confidence
  - Want close to 0
- Area Under ROC Curve (AUC)
  - 1 is perfect
  - .5 basically random
- Sensitivity: True positive rate (recall)
- Specificity: True negative rate
- Confusion Matrix
  - Accuracy for each class
- Classification Report
  - Precision, recall, f1-score, and support
- Regression Metrics
  - Mean Absolute Error (MAE)
    - Sum of absolute differences between predictions and actual values
  - Mean Squared Error (MSE)
    - Gets magnitude of error
    - taking sqrt of this converts back to original unit
  - R2 Metric
    - 0 is no fit 1 is fit
"""
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from get_data import X, y, boston_df

print('Classification Accuracy')
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = LogisticRegression(max_iter=1000)
scoring = 'accuracy'
results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
print('Accuracy:', results.mean()*100, results.std()*100)

print('Logarithmic Loss')
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = LogisticRegression(max_iter=1000)
scoring = 'neg_log_loss'
results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
print('Logloss:', results.mean()*100, results.std()*100)

print('ROC AUC')
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = LogisticRegression(max_iter=1000)
scoring = 'roc_auc'
results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
print('AUC:', results.mean(), results.std())

print('Confusion Matrix')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.33, random_state=7)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
predicted = model.predict(X_test)
matrix = confusion_matrix(y_test, predicted)
print(matrix)

print('Classification Report')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.33, random_state=7
)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
predicted = model.predict(X_test)
report = classification_report(y_test, predicted)
print(report)

print('Mean Absolute Error')
X = boston_df.values[:, :13]
y = boston_df.values[:, 13]
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = LinearRegression()
scoring = 'neg_mean_absolute_error'
results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
print('MAE:', results.mean(), results.std())

print('Mean Squared Error')
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = LinearRegression()
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
print('MSE:', results.mean(), results.std())

print('R2 metric')
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
model = LinearRegression()
scoring = 'r2'
results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
print('R2:', results.mean(), results.std())
