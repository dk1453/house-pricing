import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import GridSearchCV
import catboost as cb

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

model = cb.CatBoostRegressor()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
pd.DataFrame({
    "Id": X_test["Id"],
    "SalePrice": y_pred,
}).to_csv("cb_baselince.csv", index=False)