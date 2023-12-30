import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
# get some basic information of the missing value
# check the rows containing missing values
def MissingValueInRows(df, show=False):
    "Create an empty DataFrame to store counts of NaN values"
    index_rows_nan = pd.DataFrame(columns=['Row_Index', 'NaN_Count'])
    "Iterate through rows by index"
    for i in range(df.shape[0]):
        if df.iloc[i, :].isna().sum() != 0:
            row_index = i + 1
            nan_count = df.iloc[i, :].isna().sum()
            index_rows_nan = index_rows_nan._append({'Row_Index': row_index, 'NaN_Count': nan_count}, ignore_index=True)
    if show == True:
        pd.set_option('display.max_rows', None)
        pd.set_option('display.expand_frame_repr', False)  # Prevent line-wrapping
        print(index_rows_nan)
        pd.reset_option('display.max_rows')
        pd.reset_option('display.expand_frame_repr')
    else:
        print("The number of rows with missing values is {}.".format(index_rows_nan.shape[0]))

def MissingValueInColumns(df, show=False):
    "check the columns containing missing values"
    "Create an empty DataFrame to store counts of NaN values"
    counts_of_nan = pd.DataFrame(columns=['Column_Name', 'NaN_Count'])
    "Iterate through columns by index"
    for i in range(df.shape[1]):
        column_name = df.columns[i]
        nan_count = df.iloc[:, i].isna().sum()
        counts_of_nan = counts_of_nan._append({'Column_Name': column_name, 'NaN_Count': nan_count}, ignore_index=True)

    "Use boolean indexing to filter rows where the number of missing value is not zero"
    Festures_with_missing_values = counts_of_nan[counts_of_nan['NaN_Count'] != 0]

    if show == True:

        "Display the DataFrame with counts of NaN value"
        "Set Pandas display options to show all columns without truncation"
        pd.set_option('display.max_rows', None)
        pd.set_option('display.expand_frame_repr', False)  # Prevent line-wrapping

        "Print the DataFrame with all columns"
        print(counts_of_nan)

        "Reset the display options to the default (optional)"
        pd.reset_option('display.max_rows')
        pd.reset_option('display.expand_frame_repr')
    else :
        print("Festures with missing values:")
        print(Festures_with_missing_values)





# test
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

# MissingValueInRows(X_train)
# MissingValueInColumns(X_train)
# MissingValueInColumns(X_train,True)
