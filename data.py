import pandas as pd


filename = 'data/pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin',
         'test', 'mass', 'pedi', 'age', 'class']
df = pd.read_csv(filename, names=names)
