import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

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

# parameter tuning
parameters = {'max_depth': [5, 10, 15, 20, 25],
    'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15]}

clf = xgb.XGBRegressor()

grid_search = GridSearchCV(clf, param_grid=parameters, scoring='roc_auc', cv=5)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)


# model = xgb.XGBRegressor()
#
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# pd.DataFrame({
#     "Id": X_test["Id"],
#     "SalePrice": y_pred,
# }).to_csv("xgb_baselince.csv", index=False)