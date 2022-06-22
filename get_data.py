import pandas as pd


filename = 'csv_files/pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin',
         'test', 'mass', 'pedi', 'age', 'class']
df = pd.read_csv(filename, names=names)
X = df.values[:, :8]
y = df.values[:, 8]
