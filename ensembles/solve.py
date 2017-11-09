import pandas as pd
import numpy as np
from pygments.lexer import include
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge
from itertools import product
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import roc_auc_score


train = pd.read_csv('train.csv', index_col = 'Id')

X, y = train.drop('SalePrice', axis=1), train.SalePrice.copy()

def rmsle(estimador, X, y):

    ''' Método para calcular o erro utilizando RMSLE

    :param estimador: eh um modelo jah treinado
    :param X: features
    :param y: alvo
    :return: erro
    '''

    p = estimador.predict(X)
    return np.sqrt(mean_squared_error(np.log1p(y), np.log1p(p)))

def mean_sqrt(estimador, X,y):
    p = estimador.predict(X)
    return np.sqrt(mean_squared_error(y, p))


def rmsle_log_y(estimator, X, y):

    p = estimator.predict(X)
    return np.sqrt(mean_squared_error(y, p))


def rmsle_sqrt_y(estimator, X, y):

    p = estimator.predict(X)
    y = np.power(y, 2)
    p = np.power(p, 2)
    return np.sqrt(mean_squared_error(np.log1p(y), np.log1p(p)))

kf = KFold(n_splits=5, shuffle=True, random_state=1)

# Separando variaveis categoricas de numericas

X1 = X.select_dtypes(include=[np.number])

# Procura por dados N/A e substitui por -1
X1 = X1.fillna(-1)

# Treinando um modelo de arvore com somente as variaveis numericas

#
# model = RandomForestRegressor(n_estimators=1000, random_state=0)
# error = cross_val_score(model, X1, y, cv=kf, scoring=rmsle).mean()
# print('Dims', X1.shape)
# print('RMSLE: ', error)


# Faz a codificacao das colunas por classes
X2 = X.copy()

for col in X2.columns:

    if X2[col].dtype == object:
        encoded = LabelEncoder()
        X2[col] = encoded.fit_transform(X[col].fillna('Missing'))



X2.fillna(-1, inplace=True)

# model = RandomForestRegressor(n_estimators=1000, random_state=0)
# error = cross_val_score(model, X2, y, cv=kf, scoring=rmsle).mean()
# print('Dims', X2.shape)
# print('RMSLE: ', error)


# Faz a codificacao em classes e depois faz o one hot encoding

X3 = X.copy()

cats = []

for col in X3.columns:

    if X3[col].dtype == object:
        # Expande a coluna
        X3 = X3.join(pd.get_dummies(X3[col], prefix=col), how='left')
        X3.drop(col, axis=1, inplace=True)


# print("-------Random Forest--------")
X3.fillna(-1, inplace=True)
# model = RandomForestRegressor(n_estimators=1000, random_state=0)
# #error = cross_val_score(model, X3, y, cv=kf, scoring=rmsle).mean()
# print('Dims', X3.shape)
# #print('RMSLE: ', error)

''' Uma maneira interessante de criar diversidade, e às vezes até obter uma melhor performance, 
    num caso de regressão, é transformar a variável que estamos tentando prever. Neste caso 
    testaremos duas transformações: logaritmo e raiz quadrada.
'''
#print("Log")

# model = RandomForestRegressor(n_estimators=1000, random_state=0)
# error = cross_val_score(model, X1, np.log1p(y), cv=kf, scoring=rmsle_log_y).mean()
# print('RF, X1, log-target RMSLE:', error)
#
# print("Raiz Quadrada")
# model = RandomForestRegressor(n_estimators=1000, random_state=0)
# error = cross_val_score(model, X2, np.log1p(y), cv=kf, scoring=rmsle_log_y).mean()
# print('RF, X2, log-target RMSLE:', error)
#
#
# print("-------Gradient Boosting--------")
# print("Log")
# model = GradientBoostingRegressor(random_state=0)
# error = cross_val_score(model, X1, np.log1p(y), cv=kf, scoring=rmsle_log_y).mean()
# print('GBM, X1, log-target RMSLE:', error)
#
# model = GradientBoostingRegressor(random_state=0)
# error = cross_val_score(model, X2, np.log1p(y), cv=kf, scoring=rmsle_log_y).mean()
# print('GBM, X2, log-target RMSLE:', error)
#
# print("Raiz Quadrada")
# model = GradientBoostingRegressor(random_state=0)
# error = cross_val_score(model, X1, np.sqrt(y), cv=kf, scoring=rmsle_sqrt_y).mean()
# print('GBM, X1, sqrt-target RMSLE:', error)
#
# model = GradientBoostingRegressor(random_state=0)
# error = cross_val_score(model, X2, np.sqrt(y), cv=kf, scoring=rmsle_sqrt_y).mean()
# print('GBM, X2, sqrt-target RMSLE:', error)



# print("-------ADA Boost--------")
# print("Log")
# model = AdaBoostRegressor(n_estimators=1000,random_state=0)
# error = cross_val_score(model, X1, np.log1p(y), cv=kf, scoring=rmsle_log_y).mean()
# print('GBM, X1, log-target RMSLE:', error)
#
# model = AdaBoostRegressor(n_estimators=1000,random_state=0)
# error = cross_val_score(model, X2, np.log1p(y), cv=kf, scoring=rmsle_log_y).mean()
# print('GBM, X2, log-target RMSLE:', error)
#
# print("Raiz Quadrada")
# model = AdaBoostRegressor(n_estimators=1000,random_state=0)
# error = cross_val_score(model, X1, np.sqrt(y), cv=kf, scoring=rmsle_sqrt_y).mean()
# print('GBM, X1, sqrt-target RMSLE:', error)
#
# model = AdaBoostRegressor(n_estimators=1000,random_state=0)
# error = cross_val_score(model, X2, np.sqrt(y), cv=kf, scoring=rmsle_sqrt_y).mean()
# print('GBM, X2, sqrt-target RMSLE:', error)


#Fazendo o ensemble
kf_out = KFold(n_splits=5, shuffle=True, random_state=1)
kf_in = KFold(n_splits=5, shuffle=True, random_state=2)

cv_mean = []

# Separa em grupos para treino e teste
for fold, (tr, ts) in enumerate(kf_out.split(X, y)):

    X1_train, X1_test = X1.iloc[tr], X1.iloc[ts]
    X2_train, X2_test = X2.iloc[tr], X2.iloc[ts]

    y_train, y_test = y.iloc[tr], y.iloc[ts]

    models = [GradientBoostingRegressor(random_state=0), RandomForestRegressor(random_state=0, n_estimators=1000)]
    targets = [np.log1p, np.sqrt]
    feature_sets = [(X1_train, X1_test), (X2_train, X2_test)]

    predictions_cv = []
    predictions_test = []

    # para cada model, para cada target, para cada feature set. mesma coisa que 3 for encadeados
    for model, target, feature_set in product(models, targets, feature_sets):

        predictions_cv.append(cross_val_predict(model, feature_set[0], target(y_train), cv=kf_in).reshape(-1,1))

        # feture_set[0] -> X1_train ou X2_train
        model.fit(feature_set[0], target(y_train))
        ptest = model.predict(feature_set[1])
        predictions_test.append(ptest.reshape(-1,1))

    predictions_cv = np.concatenate(predictions_cv, axis=1)
    predictions_test = np.concatenate(predictions_test, axis=1)

    #
    stacking = Ridge()
    # treinando o modelo
    stacking.fit(predictions_cv, np.log1p(y_train))
    error = rmsle_log_y(stacking, predictions_test, np.log1p(y_test))
    cv_mean.append(error)
    print('RMSLE Fold %d - RMSLE %.4f' % (fold, error))

print('RMSLE CV5 %.4f' % np.mean(cv_mean))