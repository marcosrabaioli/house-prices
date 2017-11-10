import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from scipy import stats
import warnings
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from itertools import product
from sklearn.preprocessing import LabelEncoder


warnings.filterwarnings('ignore')

ds_test = pd.read_csv('test.csv')

ds_train = pd.read_csv('train.csv')

# Tratamendo dos dados de treino

total = ds_train.isnull().sum().sort_values(ascending=False)
percent = (ds_train.isnull().sum()/ds_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

print("------Tratamendo dos dados de treino:--------")
print("Dados ausentes:")
print(missing_data)

# Deletando colunas com mais de 5% de dados ausentes
ds_train = ds_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
ds_train = ds_train.drop(ds_train.loc[ds_train['Electrical'].isnull()].index)

ds_test= ds_test.drop((missing_data[missing_data['Total'] > 1]).index,1)


print("Dados ausentes no set de treino: ", ds_train.isnull().sum().max())
print("Shape de traino: ", ds_train.shape)

print("Dados ausentes no set de test: ", ds_test.isnull().sum().max())
print("Shape de teste: ", ds_test.shape)

# Eliminando dados mentirosos do set de treino
ds_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
ds_train = ds_train.drop(ds_train[ds_train['Id'] == 1299].index)
ds_train = ds_train.drop(ds_train[ds_train['Id'] == 524].index)

# Aplicando transformacao logaritmica em SalePrice
# Curtose de SalePrice eh positiva, entao aplica-se log para normalizar
ds_train['SalePrice'] = np.log(ds_train['SalePrice'])

# Extraindo SalePrice de set de treino e juntando set de treino e teste para simplificar a manipulacao dos conjuntos
y_train = ds_train.SalePrice.copy()
X_set = pd.concat([ds_train.drop('SalePrice', axis=1), ds_test], ignore_index=True, keys=['train', 'test'])

del ds_train
del ds_test

print("Shape do conjunto total: ", X_set.shape)

# Aplicando transformacao log em GrLivArea
X_set['GrLivArea'] = np.log(X_set['GrLivArea'])

# Criando nova variavel para TotalBsmtSF (tem ou nao BsmtSF)
# Se area > 0 HasBsmt = 1, para area == 0 it HasBsmt 0
X_set['HasBsmt'] = pd.Series(len(X_set['TotalBsmtSF']), index=X_set.index)
X_set['HasBsmt'] = 0
X_set.loc[X_set['TotalBsmtSF']>0,'HasBsmt'] = 1
#transform data
X_set.loc[X_set['HasBsmt']==1,'TotalBsmtSF'] = np.log(X_set['TotalBsmtSF'])

# Separando conjunto de dados
columns = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt', 'HasBsmt']

# Conjunto de dados de alta correlacao
X_set_hi_correlation = pd.DataFrame(X_set[columns[0]])

for col in columns:

    # Expande a coluna
    if col != columns[0]:
        X_set_hi_correlation = X_set_hi_correlation.join(X_set[col], how='left')

print("Shape conjuntos de dados de alta correlacao: ", X_set_hi_correlation.shape)






