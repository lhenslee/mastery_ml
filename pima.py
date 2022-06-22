import numpy
import pandas as pd
from matplotlib import pyplot
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Binarizer, MinMaxScaler, StandardScaler, Normalizer
numpy.set_printoptions(precision=3)

"""Chapter 5
- Peek at your data
- Dimensions of data
- Data types
- Class distribution
- Data Summary
- Correlations
- Skewness
"""
filename = 'data/pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin',
         'test', 'mass', 'pedi', 'age', 'class']
df = pd.read_csv(filename, names=names)
print('df head')
print(df.head())
print('Class distribution')
print(df.groupby('class').size())
print('Description')
print(df.describe())
print('Correlation: 0 means uncorrelated, thats what we want')
print(df.corr(method='pearson'))
print('Skew: Data is mostly on one side of normal')
print(df.skew())

"""Chapter 6: Data Visualization
- Univariate Plots
  - Histogram, density: Guassian, skewed, or exponential?
  - Box: Good to check skewed. Dots are outliers
- Multivariate Plots
  - Correlation Matrix Plot
    - Linear and logistic regression have poor performance with highly correlated input variables
  - Scatter Plot Matrix
    - Try to remove structured relationships (can draw a line through)
"""
# Univariate plots
#df.hist(figsize=(15, 9))
# pyplot.show()
#df.plot(kind='density', subplots=True, layout=(3, 3), sharex=False, figsize=(15, 9))
# pyplot.show()
#df.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False, figsize=(15, 9))
# pyplot.show()

# Multivariate plots


def correlation_matrix():
    correlations = df.corr()
    fig = pyplot.figure(figsize=(15, 9))
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = numpy.arange(0, 9, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    pyplot.show()


def scatter_plot_matrix():
    pd.plotting.scatter_matrix(df, figsize=(15, 9))
    pyplot.show()
# scatter_plot_matrix()


"""Chapter 7: Prepare data
- Rescale data
  - All data has range of 0-1
- Standardization: Transform to a standard Gaussian distribution
  - Mean value of 0 and standard deviation of 1
- Normalize data
  - Good for sparse datasets
  - Weight
  - Row length of 1
"""
array = df.values
X = array[:, :8]
y = array[:, 8]
print('Rescaled')
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)
print(rescaledX[:5])
print('Standardized')
scaler = StandardScaler().fit(X)
standardX = scaler.transform(X)
print(standardX[:5])
print('Normalized')
scaler = Normalizer().fit(X)
normalizedX = scaler.transform(X)
print(normalizedX[:5])
print('Binarized')
scaler = Binarizer(threshold=0.0).fit(X)
binaryX = scaler.transform(X)
print(binaryX[:5])

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
print('chi2 univariate selection of 4')
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, y)
print(fit.scores_)
features = fit.transform(X)
print(features[:5])
print('Recursive Feature Elimination')
model = LogisticRegression()
rfe = RFE(model)
fit = rfe.fit(rescaledX, y)
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
