"""Chapter 8: Feature selection
- Reduces overfitting (redundant data)
- Improvess Accuracy (less misleading data)
- Reduces training time
- Univariate Selection
  - chi-squared for non-negative features to select k features
- Recursive Feature Elimination
  - Use any model with a fit
- Principal Component Analysis
  - Linear algebra to transform dataset into compressed form
"""
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from get_data import df, names

X = df.values[:, :8]
y = df.values[:, 8]


print('chi2 univariate selection of 4')
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, y)
print(fit.scores_)
features = fit.transform(X)
print(features[:5])
print('Recursive Feature Elimination')
model = LogisticRegression()
rfe = RFE(model)
fit = rfe.fit(X, y)
print('Num Features:', fit.n_features_)
print('Selected Features:', fit.support_)
print(names)
print('Feature Ranking:', fit.ranking_)
print('Principal Component Analysis')
pca = PCA(n_components=3)
fit = pca.fit(X)
print("Explained Variance:", fit.explained_variance_ratio_)
print(fit.components_)
print('Feature importance with ExtraTeesClassifer')
model = ExtraTreesClassifier()
model.fit(X, y)
print(names)
print(model.feature_importances_)
