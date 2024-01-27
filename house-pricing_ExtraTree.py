import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor


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
# # round 1
# start = time.time()
# # parameter space
# parameter_space = {
#     "n_estimators": range(30,71,5),
#     "min_samples_split": range(2,16),
#     "max_features":['sqrt','log2']
# }
# ExtraTree_0 = ExtraTreesRegressor(random_state=2)
# grid_ExtraTree_0 = GridSearchCV(ExtraTree_0, parameter_space, n_jobs=-1)
# grid_ExtraTree_0.fit(X_train,y_train)
# print(time.time() - start)
# print(grid_ExtraTree_0.best_score_)
# print(grid_ExtraTree_0.best_params_)
# # 13.894102096557617
# # 0.8554840038699624
# # {'max_features': 'sqrt', 'min_samples_split': 6, 'n_estimators': 40}

# round 2
# start = time.time()
# # parameter space
# parameter_space = {
#     "n_estimators": range(20,61,5),
#     "min_samples_split": range(3,15),
#     "max_features":['sqrt','log2']
# }
# ExtraTree_0 = ExtraTreesRegressor(random_state=2)
# grid_ExtraTree_0 = GridSearchCV(ExtraTree_0, parameter_space, n_jobs=-1)
# grid_ExtraTree_0.fit(X_train,y_train)
# print(time.time() - start)
# print(grid_ExtraTree_0.best_score_)
# print(grid_ExtraTree_0.best_params_)
# # 10.50204610824585
# # 0.8554840038699624
# # {'max_features': 'sqrt', 'min_samples_split': 6, 'n_estimators': 40}

# round 3
# start = time.time()
# # parameter space
# parameter_space = {
#     "n_estimators": range(35,46),
#     "min_samples_split": range(3,15),
#     "max_features":['sqrt','log2']
# }
# ExtraTree_0 = ExtraTreesRegressor(random_state=2)
# grid_ExtraTree_0 = GridSearchCV(ExtraTree_0, parameter_space, n_jobs=-1)
# grid_ExtraTree_0.fit(X_train,y_train)
# print(time.time() - start)
# print(grid_ExtraTree_0.best_score_)
# print(grid_ExtraTree_0.best_params_)
# # 14.542773962020874
# # 0.8557462895592931
# # {'max_features': 'sqrt', 'min_samples_split': 6, 'n_estimators': 38}

# model fitting
model = ExtraTreesRegressor(n_estimators=38,min_samples_split=6,max_features='sqrt')

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
pd.DataFrame({
    "Id": X_test["Id"],
    "SalePrice": y_pred,
}).to_csv("ExtraTree_gridsearch.csv", index=False)