import pandas as pd


filename = 'csv_files/pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin',
         'test', 'mass', 'pedi', 'age', 'class']
df = pd.read_csv(filename, names=names)
X = df.values[:, :8]
y = df.values[:, 8]

boston_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
                'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
boston_df = pd.read_csv('csv_files/housing.csv', delim_whitespace=True)
