import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression



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

# Define base estimators
estimators = [
    ('lgbm', lgb.LGBMRegressor(
        random_state = 12,
        max_depth = np.arange(10, 50).tolist()[16],
        num_leaves = np.arange(10, 50).tolist()[4],
        n_estimators = np.arange(10, 100).tolist()[73],
        boosting_type = ['gbdt', 'goss'][0],
        colsample_bytree = 0.36542426350386015,
        learning_rate = 0.09023431304252874,
        reg_alpha = 0.3829318653803218,
        reg_lambda = 0.05553726484225459)),
    ('xgb', xgb.XGBRegressor()),
    ('cb', cb.CatBoostRegressor())
]

# # Create a stacking regressor
# stacking_regressor = StackingRegressor(
#     estimators=estimators,
#     final_estimator=cb.CatBoostRegressor()
# )

# Create the second stacking regressor using logistic

stacking_regressor = StackingRegressor(
    estimators=estimators,
    final_estimator= LinearRegression()
)

# Train and use the stacking regressor
stacking_regressor.fit(X_train, y_train)
y_pred = stacking_regressor.predict(X_test)
pd.DataFrame({
    "Id": X_test["Id"],
    "SalePrice": y_pred,
}).to_csv("stacking_baseline.csv", index=False)