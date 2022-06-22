# Python Project Template

# 1. Prepare Problem
# a) Load libraries
from matplotlib import pyplot as plt
import pandas as pd
# b) Load dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
df = pd.read_csv('iris.csv', names=names)

# 2. Summarize Data
# a) Descriptive statistics
print(df.shape)
print(df.head(20))
print(df.describe())
# b) Data visualizations
df.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
#plt.show()
df.hist()
#plt.show()
pd.plotting.scatter_matrix(df)
#plt.show()

# 3. Prepare Data
# a) Data Cleaning
# b) Feature Selection
# c) Data Transforms

# 4. Evaluate Algorithms
# a) Split-out validation dataset
# b) Test options and evaluation metric
# c) Spot Check Algorithms
# d) Compare Algorithms

# 5. Improve Accuracy
# a) Algorithm Tuning
# b) Ensembles

# 6. Finalize Model
# a) Predictions on validation dataset
# b) Create standalone model on entire training dataset
# c) Save model for later use
