"""
Chapter 5
- Peek at your data
- Dimensions of data
- Data types
- Class distribution
- Data Summary
- Correlations
- Skewness
"""
from get_data import df


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
