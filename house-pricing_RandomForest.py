import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from function import MissingValueInRows, MissingValueInColumns

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

# print(X_train.head())

# MissingValueInRows(X_train) # 339 missing value
print("---------------------------")
# MissingValueInColumns(X_train)# 259+81+8=348>339, there are samples with more than one missing value
print("Training data has {} samples and {} features".format(X_train.shape[0], X_train.shape[1]))
# Now we know the features with missing value and the corresponding number of missing value. It's hard to detect if the missing value is random so I have to impute
# get info of features with missing values
print(X_train[['LotFrontage','MasVnrArea','GarageYrBlt']].describe())
# LotFrontage: the missing value can be imputed with the average of the neighbors, 259 missing
X_train['LotFrontage'] = Xy_train.groupby('Neighborhood')['LotFrontage'].transform(lambda x:x.fillna(x.median()))
# MasVnrArea is Masonry veneer area in square feet, we can see there are lots of 0 here, so NaN value can be treated as 0, 8 missing
X_train['MasVnrArea'] = X_train['MasVnrArea'].fillna(0)
# GarageYrBlt is Year garage was built, as it is a time-related feature so filling with 0 is unreasonable, mode makes the most sense,  81 missing
mode_year = X_train['GarageYrBlt'].mode().values[0]
X_train['GarageYrBlt'].fillna(mode_year, inplace=True)

# check if there is still missing values in both train dataset and test dataset
print("The number of missing values in train data set is {}.".format(X_train.isna().sum().sum()))
print('------')
print("The number of missing values in test data set is {}.".format(X_test.isna().sum().sum()))
print('------')

# MissingValueInRows(X_test)
# MissingValueInColumns(X_test)

# repeat the process for the test dataset
X_test['LotFrontage'] = Xy_train.groupby('Neighborhood')['LotFrontage'].transform(lambda x:x.fillna(x.median()))
X_test['MasVnrArea'] = X_test['MasVnrArea'].fillna(0)
X_test['GarageYrBlt'].fillna(X_test['GarageYrBlt'].mode().values[0], inplace=True)

# analyze the features with missing values
pd.set_option('display.max_columns', None)
print(X_train[['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','GarageCars','GarageArea']].describe())
pd.reset_option('display.max_columns')
# the mins of these features are all 0, so it is reasonable to assume the missing values are all 0
X_test['BsmtFinSF1'] = X_test['BsmtFinSF1'].fillna(0)
X_test['BsmtFinSF2'] = X_test['BsmtFinSF2'].fillna(0)
X_test['BsmtUnfSF'] = X_test['BsmtUnfSF'].fillna(0)
X_test['TotalBsmtSF'] = X_test['TotalBsmtSF'].fillna(0)
X_test['BsmtFullBath'] = X_test['BsmtFullBath'].fillna(0)
X_test['BsmtHalfBath'] = X_test['BsmtHalfBath'].fillna(0)
X_test['GarageCars'] = X_test['GarageCars'].fillna(0)
X_test['GarageArea'] = X_test['GarageArea'].fillna(0)

# double check
print("------------------------")
print("The number of missing values in test data set is:")
print(sum(X_test.isna().sum()))


# model fitting
model = RandomForestRegressor()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
pd.DataFrame({
    "Id": X_test["Id"],
    "SalePrice": y_pred,
}).to_csv("RandomForest_baseline.csv", index=False)