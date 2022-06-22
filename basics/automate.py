"""
Chapter 14: Automate Machine Learning Workflows with pipelines
- Data prep and model pipeline
  - Standarize data
  - Learn linear discriminant analasys model
- Feature extraction and model pipeline
"""
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from get_data import X, y

print('Stardize Data and Model Pipeline')
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('lda', LinearDiscriminantAnalysis()))
model = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())

print('Feature Extraction and Model Pipeline')
features = [
    ('pca', PCA(n_components=3)),
    ('select_best', SelectKBest(k=6))
]
feature_union = FeatureUnion(features)
estimators = [
    ('feature_union', feature_union),
    ('logistic', LogisticRegression(max_iter=1000))
]
model = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())
