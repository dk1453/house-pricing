import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import cross_val_score
import lightgbm as lgb
from hyperopt import hp, fmin, tpe, Trials, partial
from hyperopt.early_stop import no_progress_loss

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

# # parameter tuning
# # Define the hyperparameter search space
# LGBM_params_space = {'max_depth': hp.choice('max_depth', np.arange(10, 50).tolist()),
#                      'num_leaves': hp.choice('num_leaves', np.arange(10, 50).tolist()),
#                      'n_estimators': hp.choice('n_estimators', np.arange(10, 100).tolist()),
#                      'boosting_type': hp.choice('boosting_type', ['gbdt', 'goss']),
#                      'colsample_bytree': hp.uniform('colsample_bytree', 0.2, 1.0),
#                      'learning_rate': hp.uniform('learning_rate', 0.01, 0.1),
#                      'reg_alpha': hp.uniform('reg_alpha', 0.01, 0.5),
#                      'reg_lambda': hp.uniform('reg_lambda', 0.01, 0.5)}
#
# # Define the objective function
# def hyperopt_lgbm(params):
#     # Read Parameters
#     max_depth = params['max_depth']
#     num_leaves = params['num_leaves']
#     n_estimators = params['n_estimators']
#     boosting_type = params['boosting_type']
#     colsample_bytree = params['colsample_bytree']
#     learning_rate = params['learning_rate']
#     reg_alpha = params['reg_alpha']
#     reg_lambda = params['reg_lambda']
#     # Instantiate the model
#     lgbm = lgb.LGBMRegressor(
#         random_state = 12,
#         max_depth = max_depth,
#         num_leaves = num_leaves,
#         n_estimators = n_estimators,
#         boosting_type = boosting_type,
#         colsample_bytree = colsample_bytree,
#         learning_rate = learning_rate,
#         reg_alpha = reg_alpha,
#         reg_lambda = reg_lambda)
#     # Output the results of cross-validation
#     res = -cross_val_score(lgbm, X_train, y_train).mean()
#
#     return res
#
# # Define Optimization Function
# def param_hyperopt_lgbm(max_evals):
#     params_best = fmin(fn = hyperopt_lgbm,
#                        space = LGBM_params_space,
#                        algo = tpe.suggest,
#                        max_evals = max_evals)
#
#     return params_best
#
# #  test
# lgbm_params_best = param_hyperopt_lgbm(1000)
# print(lgbm_params_best)

model = lgb.LGBMRegressor(
        random_state = 12,
        max_depth = np.arange(10, 50).tolist()[16],
        num_leaves = np.arange(10, 50).tolist()[4],
        n_estimators = np.arange(10, 100).tolist()[73],
        boosting_type = ['gbdt', 'goss'][0],
        colsample_bytree = 0.36542426350386015,
        learning_rate = 0.09023431304252874,
        reg_alpha = 0.3829318653803218,
        reg_lambda = 0.05553726484225459)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
pd.DataFrame({
    "Id": X_test["Id"],
    "SalePrice": y_pred,
}).to_csv("lgb_baseline.csv", index=False)
