from settings import *
import random

def createMCAR_single(lst, pct_miss, seed=seed_num):
    """
    Take a list of values and randomly mask them as numpy NaN.
    :param lst: a list of values
    :param pct_miss: percentage of value missing
    :param seed: default is set in the settings.py
    :return: a list of values with np.nan
    """
    random.seed(seed)
    idx_na = random.sample(range(len(lst)), int(round(pct_miss*len(lst))))
    lst = [np.nan if i in idx_na else lst[i] for i in range(len(lst))]
    return np.array([lst])

# test case
temp = createMCAR_single(df_nona['age'].values.tolist(), 0.01)

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
temp = createMCAR_df(df_nona, ["age", "sysBP", "diaBP"], 0.01)

def compute_imputation_MAE_pct(df_to_impute, name_data_to_impute, df_imputed, name_impute_method, df_original=df_nona):

    """
    This function calculate the metric to evaluate imputation performance.
    For every missing entries, calculate the percentage difference between the original and imputed values.
    The error score for the entire imputed dataset is the sum of absolute percentage difference, divided by the total missing entries
    Return a data frame with 4 cols: the version of data used for imputation, imputation method, error score, and a dict that contains the detailed pct_diff for each var.

    :param df_to_impute: data frame that was used for imputation
    :param name_data_to_impute: a str, the version name of the data frame used for imputation. Format: na_pct_5-age; na_pct_5-batch
    :param df_imputed: data frame that was imputed
    :param name_impute_method: a str, the version name of imputation methodology. Format: rf
    :param df_original: the original complete data frame
    :return: a dataframe that summarize the result
    """

    temp = df_to_impute.isnull().sum()
    cols_imputed = temp[temp>0].reset_index()['index'].to_list()
    dict_result = {}
    total_pct_diff = 0
    n = df_to_impute.isna().sum().sum()
    idx_range = range(df_original.shape[0])

    for var in cols_imputed:

        idx_na = df_to_impute[df_to_impute[var].isna()].index.tolist()
        diff = df_original[var] - df_imputed[var]
        diff = [diff[i] for i in idx_range if i in idx_na]
        origin = [df_original[var][i] for i in idx_range if i in idx_na]

        pct_diff = pd.Series(diff)/pd.Series(origin)
        sum_abs_diff = abs(pct_diff).sum()

        total_pct_diff = total_pct_diff + sum_abs_diff
        dict_result[var] = pct_diff
    print(total_pct_diff/n)
    df_result = pd.DataFrame({"data_to_impute": [name_data_to_impute],
                              "impute_method": [name_impute_method],
                              "percentage_MAE": [total_pct_diff/n]})

    return df_result

# test
temp = compute_imputation_MAE_pct(temp, "testing", df_nona, "cheating")



