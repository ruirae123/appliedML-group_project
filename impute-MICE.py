from settings import *
from utils import *
from fancyimpute import IterativeImputer

def get_imputeResult_MICE(f_path, f_name):
    """
    Use MICE for imputation and get results.
    :param f_path: path to the data to be imputed
    :param f_name: name of the data file
    :return: a data frame that summarize the imputation performance
    """
    df_toImpute = pd.read_csv(f_path)

    mice_imputer = IterativeImputer()
    df_complete = mice_imputer.fit_transform(df_toImpute)
    df_complete = pd.DataFrame(data=df_complete, columns=df_toImpute.columns.tolist())

    df_result = compute_imputation_MAE(df_toImpute, f_name, df_complete, 'MICE')

    return  df_result

df_result = pd.DataFrame()
for f_name in os.listdir(path_data_processed):
    temp = get_imputeResult_MICE(f_path=os.path.join(path_data_processed, f_name), f_name=f_name.split('.')[0])
    df_result = pd.concat([df_result, temp], axis=0)
