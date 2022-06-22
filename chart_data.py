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
from matplotlib import pyplot as plt
import numpy as np
from get_data import names, df, pd
# Univariate plots
#df.hist(figsize=(15, 9))
# plt.show()
#df.plot(kind='density', subplots=True, layout=(3, 3), sharex=False, figsize=(15, 9))
# plt.show()
#df.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False, figsize=(15, 9))
# plt.show()

# Multivariate plots


def correlation_matrix():
    correlations = df.corr()
    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, 9, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    plt.show()


def scatter_plot_matrix():
    pd.plotting.scatter_matrix(df, figsize=(15, 9))
    plt.show()


scatter_plot_matrix()
