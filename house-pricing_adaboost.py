import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostRegressor


# feature engineering
Xy_train = pd.read_csv('train.csv')
X_test = pd.read_csv('test.csv')
Xy_all = pd.concat([Xy_train, X_test], axis=0)
cat_features = Xy_all.columns[Xy_all.dtypes == 'object']
ordinal_encoder = OrdinalEncoder(
    dtype=np.int32,
    handle_unknown='use_encoded_value',
    unknown_value=-1,
    encoded_missing_value=-1,
).set_output(transform="pandas")
Xy_all[cat_features] = ordinal_encoder.fit_transform(Xy_all[cat_features])
X_test = Xy_all[Xy_all["SalePrice"].isna()].drop(columns=["SalePrice"])
Xy_train = Xy_all[~Xy_all["SalePrice"].isna()]
X_train = Xy_train.drop(columns=["SalePrice"])
y_train = Xy_train["SalePrice"]

# fill the missing value for X_train, details are in RandomForest
X_train['LotFrontage'] = Xy_train.groupby('Neighborhood')['LotFrontage'].transform(lambda x:x.fillna(x.median()))
X_train['MasVnrArea'] = X_train['MasVnrArea'].fillna(0)
mode_year = X_train['GarageYrBlt'].mode().values[0]
X_train['GarageYrBlt'].fillna(mode_year, inplace=True)

# fill the missing value for X_test, details are in RandomForest
X_test['LotFrontage'] = Xy_train.groupby('Neighborhood')['LotFrontage'].transform(lambda x:x.fillna(x.median()))
X_test['MasVnrArea'] = X_test['MasVnrArea'].fillna(0)
X_test['GarageYrBlt'].fillna(X_test['GarageYrBlt'].mode().values[0], inplace=True)
X_test['BsmtFinSF1'] = X_test['BsmtFinSF1'].fillna(0)
X_test['BsmtFinSF2'] = X_test['BsmtFinSF2'].fillna(0)
X_test['BsmtUnfSF'] = X_test['BsmtUnfSF'].fillna(0)
X_test['TotalBsmtSF'] = X_test['TotalBsmtSF'].fillna(0)
X_test['BsmtFullBath'] = X_test['BsmtFullBath'].fillna(0)
X_test['BsmtHalfBath'] = X_test['BsmtHalfBath'].fillna(0)
X_test['GarageCars'] = X_test['GarageCars'].fillna(0)
X_test['GarageArea'] = X_test['GarageArea'].fillna(0)


# Tuning
# instantiation
# round 1
# start = time.time()
# # parameter space
# parameter_space = {
#     "n_estimators": range(30,71,5),
#     "learning_rate": list(np.linspace(0.01,0.2,5)),
#     "loss":['linear','square','exponential']
# }
# adaboost_0 = AdaBoostRegressor(random_state=2)
# grid_adaboost_0 = GridSearchCV(adaboost_0, parameter_space, n_jobs=-1)
# grid_adaboost_0.fit(X_train,y_train)
# print(time.time() - start)
# print(grid_adaboost_0.best_score_)
# print(grid_adaboost_0.best_params_)
# # 31.04758930206299
# # 0.8038672695216909
# # {'learning_rate': 0.2, 'loss': 'exponential', 'n_estimators': 70}

# round 2
# start = time.time()
# # parameter space
# parameter_space = {
#     "n_estimators": range(50,91,5),
#     "learning_rate": list(np.linspace(0.1,0.3,5)),
#     "loss":['linear','square','exponential']
# }
# adaboost_0 = AdaBoostRegressor(random_state=2)
# grid_adaboost_0 = GridSearchCV(adaboost_0, parameter_space, n_jobs=-1)
# grid_adaboost_0.fit(X_train,y_train)
# print(time.time() - start)
# print(grid_adaboost_0.best_score_)
# print(grid_adaboost_0.best_params_)
# # 38.1412148475647
# # 0.8145291115915916
# # {'learning_rate': 0.3, 'loss': 'exponential', 'n_estimators': 90}

# round 3
# start = time.time()
# # parameter space
# parameter_space = {
#     "n_estimators": range(70,111,5),
#     "learning_rate": list(np.linspace(0.3,0.5,5)),
#     "loss":['exponential']
# }
# adaboost_0 = AdaBoostRegressor(random_state=2)
# grid_adaboost_0 = GridSearchCV(adaboost_0, parameter_space, n_jobs=-1)
# grid_adaboost_0.fit(X_train,y_train)
# print(time.time() - start)
# print(grid_adaboost_0.best_score_)
# print(grid_adaboost_0.best_params_)
# # 15.721678018569946
# # 0.8192692878344421
# # {'learning_rate': 0.45, 'loss': 'exponential', 'n_estimators': 80}

# round 4
# start = time.time()
# # parameter space
# parameter_space = {
#     "n_estimators": range(75,96,5),
#     "learning_rate": list(np.linspace(0.4,0.5,5)),
#     "loss":['exponential']
# }
# adaboost_0 = AdaBoostRegressor(random_state=2)
# grid_adaboost_0 = GridSearchCV(adaboost_0, parameter_space, n_jobs=-1)
# grid_adaboost_0.fit(X_train,y_train)
# print(time.time() - start)
# print(grid_adaboost_0.best_score_)
# print(grid_adaboost_0.best_params_)
# # 8.831088066101074
# # 0.8216370214418454
# # {'learning_rate': 0.42500000000000004, 'loss': 'exponential', 'n_estimators': 85}

# round 5
# start = time.time()
# # parameter space
# parameter_space = {
#     "n_estimators": range(80,91,5),
#     "learning_rate": list(np.linspace(0.4,0.45,5)),
#     "loss":['exponential']
# }
# adaboost_0 = AdaBoostRegressor(random_state=2)
# grid_adaboost_0 = GridSearchCV(adaboost_0, parameter_space, n_jobs=-1)
# grid_adaboost_0.fit(X_train,y_train)
# print(time.time() - start)
# print(grid_adaboost_0.best_score_)
# print(grid_adaboost_0.best_params_)
# # 6.004089832305908
# # 0.8216370214418454
# # {'learning_rate': 0.42500000000000004, 'loss': 'exponential', 'n_estimators': 85}
# # the margin is small in the 5th round, stop


model = AdaBoostRegressor(
    learning_rate=0.42500000000000004,
    loss='exponential',
    n_estimators=85
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
pd.DataFrame({
    "Id": X_test["Id"],
    "SalePrice": y_pred,
}).to_csv("adaboost_gridsearch.csv", index=False)