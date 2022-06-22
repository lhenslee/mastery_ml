"""
Chapter 7: Prepare data
- Rescale data
  - All data has range of 0-1
- Standardization: Transform to a standard Gaussian distribution
  - Mean value of 0 and standard deviation of 1
- Normalize data
  - Good for sparse datasets
  - Weight
  - Row length of 1
"""
from sklearn.preprocessing import Binarizer, MinMaxScaler, Normalizer, StandardScaler
from get_data import df

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
