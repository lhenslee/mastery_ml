"""
Chapter 17: Save models
- Finalize with pickle
- Finalize with joblib
"""
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from get_data import X, y
from pickle import dump, load

print('Save model with pickle')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.33, random_state=7)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
filename = 'models/finalized_model.sav'
dump(model, open(filename, 'wb'))

print('Load model with pickle')
loaded_model = load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)

print('Save model with joblib')
joblib.dump(model, filename)

print('Load model with joblib')
loaded_model = joblib.load(filename)
result = loaded_model.score(X_test, y_test)
print(result)
