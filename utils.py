from settings import *
import random
from sklearn.preprocessing import MinMaxScaler

def createMCAR_single(lst, pct_miss, seed=seed_num):
    """
    Take a list of values and randomly mask them as numpy NaN.
    :param lst: a list of values
    :param pct_miss: percentage of value missing
    :param seed: default is set in the settings.py
    :return: a list of values with np.nan
    """
    random.seed(seed)
    idx_na = random.sample(range(len(lst)), round(pct_miss*len(lst)))
    lst_masked = [np.nan if i in idx_na else lst[i] for i in range(len(lst))]
    return np.array([lst_masked])

# test case
temp = createMCAR_single(df_nona['age'].values.tolist(), 0.2)

def createMCAR_df(df, lst_var_names, pct_miss, seed=seed_num):
    """
    Take a dataset and randomly mask values in the specified columns as numpy NaN.
    :param df: a dataframe
    :param lst_var_names: a list of variables to mask values
    :param pct_miss: percentage of value missing
    :param seed: default is set in the settings.py
    :return: a dataframe with missing values
    """

    all_columns = df.columns
    df_temp = np.array([])

    for var in all_columns:
        if var in lst_var_names:
            t = createMCAR_single(df[var].values.tolist(), pct_miss, seed)
            if len(df_temp) == 0:
                df_temp = t
            else:
                df_temp = np.vstack((df_temp, t))
        else:
            if len(df_temp) == 0:
                df_temp = np.array([df[var].values.tolist()])
            else:
                df_temp = np.vstack((df_temp, np.array([df[var].values.tolist()])))

    df_temp = pd.DataFrame(df_temp.T, columns = all_columns)
    return df_temp

# test case
temp = createMCAR_df(df_nona, ["age", "sysBP", "diaBP"], 0.1)


def compute_imputation_MAE(df_to_impute, name_data_to_impute, df_imputed, name_impute_method, df_original=df_nona):
    """
    This function calculate the metric to evaluate imputation performance.
    For every missing entries, calculate the difference between the original and imputed normalized values.
    The error score for the entire imputed dataset is the sum of absolute difference, divided by the total missing entries
    Return a data frame with 4 cols: the version of data used for imputation, imputation method, error score, and a dict that contains the detailed pct_diff for each var.

    :param df_to_impute: data frame that was used for imputation
    :param name_data_to_impute: a str, the version name of the data frame used for imputation. Format: na_pct_5-age; na_pct_5-batch
    :param df_imputed: data frame that was imputed
    :param name_impute_method: a str, the version name of imputation methodology. Format: rf
    :param df_original: the original complete data frame
    :return: a dataframe that summarize the result
    """
    # scale it so that scaling won't be an issue
    s = MinMaxScaler()
    df_original = s.fit_transform(X=df_original.to_numpy())
    df_original = pd.DataFrame(data=df_original, columns=df_to_impute.columns.tolist())
    df_imputed = s.fit_transform(X=df_imputed.to_numpy())
    df_imputed = pd.DataFrame(data=df_imputed, columns=df_to_impute.columns.tolist())

    # setup
    temp = df_to_impute.isna().sum()
    cols_imputed = temp[temp > 0].reset_index()['index'].to_list()
    dict_result = {}
    total_diff = 0
    n = df_to_impute.isna().sum().sum()

    # calculate the difference between original and imputed
    for var in cols_imputed:
        diff = df_original[var] - df_imputed[var]
        sum_abs_diff = abs(diff).sum()
        total_diff = total_diff + sum_abs_diff
        dict_result[var] = diff

    df_result = pd.DataFrame({"data_to_impute": [name_data_to_impute],
                              "impute_method": [name_impute_method],
                              "MAE_normalized": [total_diff / n]})

    return df_result

# test
temp = compute_imputation_MAE(temp, "testing", df_nona, "cheating")

def generate_DFs_forImputation(lst_vars, name_batch, lst_miss_pct_single=[0.1, 0.15, 0.3], lst_miss_pct_batch=[0.01, 0.05, 0.1, 0.15, 0.3], df=df_nona):

    """
    This function generates dataframes that are needed for imputation.
    Return nothing, but generates csv file.

    :param lst_vars: a list of variables to impute
    :param name_batch: a str, name of the csv file that has missing values for all specified variables
    :param lst_miss_pct_single:  a list of missing percentage for dataframes with missing entries for only one variable
    :param lst_miss_pct_batch: a list of missing percentage for the dataframe with missing entires for all specified variables
    :param df: a dataframe with complete rows
    :return: None
    """

    for var in lst_vars:
        for pct_miss in lst_miss_pct_single:
            df = createMCAR_df(df_nona, [var], pct_miss)
            if 'index' in df.columns.tolist():
                df.drop(columns=['index'], inplace=True)
            df.to_csv(path_data_processed+"data_"+var+"_"+str(int(pct_miss*100))+".csv", index=False)
            print("data_"+var+"_"+str(int(pct_miss*100))+".csv"+" created!")

    for pct_miss in lst_miss_pct_batch:
        df = createMCAR_df(df_nona, lst_vars, pct_miss)
        if 'index' in df.columns.tolist():
            df.drop(columns=['index'], inplace=True)
        df.to_csv(path_data_processed+"data_"+name_batch+"_"+str(int(pct_miss*100))+".csv", index=False)
        print("data_" + name_batch + "_" + str(int(pct_miss * 100)) + ".csv" + " created!")
    return None

lst_vars = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
generate_DFs_forImputation(lst_vars, "allNumerics")

# generate data for imputation
if __name__ == "main":
    lst_vars = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
    generate_DFs_forImputation(lst_vars, "allNumerics")

