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


plot_grafics = False

warnings.filterwarnings('ignore')

df_test_pre = pd.read_csv('test.csv')

df_train = pd.read_csv('train.csv')
#check the decoration
print(df_train.columns)

#descriptive statistics summary
print('\n')
print(df_train['SalePrice'].describe())

if plot_grafics:
    sns.distplot(df_train['SalePrice'])


#skewness and kurtosis
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())


#correlation matrix
corrmat = df_train.corr()
if plot_grafics:
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    plt.show()

    #saleprice correlation matrix
    k = 10 #number of variables for heatmap
    cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
    cm = np.corrcoef(df_train[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    plt.show()

    #scatterplot
    sns.set()
    cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
    sns.pairplot(df_train[cols], size = 2.5)
    plt.show()

#missing data
print('\n')
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data)

#dealing with missing data
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
print(df_train.isnull().sum().max()) #just checking that there's no missing data missing...

df_test = pd.DataFrame(df_test_pre['Electrical'])

for col in df_train.columns:
    if col != 'Electrical' and col != 'SalePrice':
        df_test = df_test.join(df_test_pre[col], how='left')

#Out liars
#bivariate analysis saleprice/grlivarea
var = 'GrLivArea'
if plot_grafics:
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
    plt.show()
#deleting points
df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)

#bivariate analysis saleprice/grlivarea
var = 'TotalBsmtSF'
if plot_grafics:
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
    plt.show()

    #histogram and normal probability plot
    sns.distplot(df_train['SalePrice'], fit=norm)
    fig = plt.figure()
    res = stats.probplot(df_train['SalePrice'], plot=plt)


#applying log transformation
df_train['SalePrice'] = np.log(df_train['SalePrice'])

#transformed histogram and normal probability plot
if plot_grafics:
    sns.distplot(df_train['SalePrice'], fit=norm)
    fig = plt.figure()
    res = stats.probplot(df_train['SalePrice'], plot=plt)
    plt.show()

    #histogram and normal probability plot
    sns.distplot(df_train['GrLivArea'], fit=norm)
    fig = plt.figure()
    res = stats.probplot(df_train['GrLivArea'], plot=plt)
    #applying log transformation
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
df_test['GrLivArea'] = np.log(df_test['GrLivArea'])
if plot_grafics:
    #transformed histogram and normal probability plot
    sns.distplot(df_train['GrLivArea'], fit=norm)
    fig = plt.figure()
    res = stats.probplot(df_train['GrLivArea'], plot=plt)
    plt.show()

    #histogram and normal probability plot
    sns.distplot(df_train['TotalBsmtSF'], fit=norm)
    fig = plt.figure()
    res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)

#create column for new variable (one is enough because it's a binary categorical feature)
#if area>0 it gets 1, for area==0 it gets 0
df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0
df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1
#transform data
df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])

df_test['HasBsmt'] = pd.Series(len(df_test['TotalBsmtSF']), index=df_test.index)
df_test['HasBsmt'] = 0
df_test.loc[df_test['TotalBsmtSF']>0,'HasBsmt'] = 1
#transform data
df_test.loc[df_test['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_test['TotalBsmtSF'])
if plot_grafics:
    #histogram and normal probability plot
    sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm)
    fig = plt.figure()
    res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
    plt.show()

    #scatter plot
    plt.scatter(df_train['GrLivArea'], df_train['SalePrice'])
    #scatter plot
    plt.scatter(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF']>0]['SalePrice'])
    plt.show()

columns = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt', 'HasBsmt']


train_most_important = pd.DataFrame(df_train['SalePrice'])
test_1 = pd.DataFrame(df_test[columns[0]])


for col in columns:

    # Expande a coluna
    train_most_important = train_most_important.join(df_train[col], how='left')
    if col != columns[0]:
        test_1 = test_1.join(df_test[col], how='left')


X1, y1 = train_most_important.drop('SalePrice', axis=1), train_most_important.SalePrice.copy()
# Faz a codificacao em classes e depois faz o one hot encoding

X2 = df_train.copy()
X2 = X2.drop('SalePrice', axis=1)
test_2 = df_test.copy()

print(X2['Id'])
print(test_2['Id'])

SET = pd.concat([X2, test_2])

print(SET['Id'])
print(SET.shape)
print(SET.columns)

SET.to_csv('SET.csv')

for col in SET.columns:

    if SET[col].dtype == object:
        # Expande a coluna
        SET = SET.join(pd.get_dummies(SET[col], prefix=col), how='left')
        SET.drop(col, axis=1, inplace=True)

SET.fillna(-1, inplace=True)

print(SET.shape)


test_2.fillna(-1, inplace=True)

print(X1.shape)
print(X2.shape)
print(test_1.shape)
print(test_2.shape)

X1.to_csv('x1.csv')
X2.to_csv('x2.csv')
test_1.to_csv('test_data_1.csv')
test_2.to_csv('test_data_2.csv')

def rmsle(estimador, X, y):

    ''' Método para calcular o erro utilizando RMSLE

    :param estimador: eh um modelo jah treinado
    :param X: features
    :param y: alvo
    :return: erro
    '''

    p = estimador.predict(X)
    return np.sqrt(mean_squared_error(np.log1p(y), np.log1p(p)))



kf = KFold(n_splits=5, shuffle=True, random_state=1)


# Procura por dados N/A e substitui por -1
X1 = X1.fillna(-1)

# Treinando um modelo de arvore com somente as variaveis numericas


#Fazendo o ensemble
kf_out = KFold(n_splits=5, shuffle=True, random_state=1)
kf_in = KFold(n_splits=5, shuffle=True, random_state=2)

cv_mean = []

# Separa em grupos para treino e teste
for fold, (tr, ts) in enumerate(kf_out.split(X1, y1)):

    X1_train, X1_test = X1.iloc[tr], X1.iloc[ts]
    X2_train, X2_test = X2.iloc[tr], X2.iloc[ts]


    y_train, y_test = y1.iloc[tr], y1.iloc[ts]

    models = [GradientBoostingRegressor(random_state=0, n_estimators=3000, learning_rate=0.1, loss='lad', max_depth=3),
              RandomForestRegressor(random_state=0, n_estimators=3000)]
    feature_sets = [(X1_train, X1_test), (X2_train, X2_test)]

    predictions_cv = []
    predictions_test = []

    # para cada model, para cada target, para cada feature set. mesma coisa que 3 for encadeados
    for model, feature_set in product(models, feature_sets):

        predictions_cv.append(cross_val_predict(model, feature_set[0], y_train, cv=kf_in).reshape(-1,1))

        # feture_set[0] -> X1_train ou X2_train
        model.fit(feature_set[0], y_train)
        ptest = model.predict(feature_set[1])
        predictions_test.append(ptest.reshape(-1,1))

    predictions_cv = np.concatenate(predictions_cv, axis=1)
    predictions_test = np.concatenate(predictions_test, axis=1)

    #
    stacking = Ridge()
    # treinando o modelo
    stacking.fit(predictions_cv, y_train)
    error = rmsle(stacking, np.exp(predictions_test), np.exp(y_test))
    cv_mean.append(error)
    print('RMSLE Fold %d - RMSLE %.4f' % (fold, error))

print('RMSLE CV5 %.4f' % np.mean(cv_mean))

# Prevendo os dados de teste para submeter ao kaggle
test_1 = test_1.fillna(-1)
predictions_model = []
for model, feature in product(models, [test_1, test_2]):
    ptest = model.predict(feature)
    predictions_model.append(ptest.reshape(-1, 1))

predictions_models = np.concatenate(predictions_model, axis=1)
predictions_submit = stacking.predict(predictions_models)

df_submit = pd.DataFrame.from_items([('SalePrice', np.exp(predictions_submit))])

print(df_submit)

df_submit.to_csv('submit.csv')


