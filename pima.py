import numpy
import pandas as pd
from matplotlib import pyplot
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Binarizer, MinMaxScaler, StandardScaler, Normalizer
numpy.set_printoptions(precision=3)
