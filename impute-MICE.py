from settings import *
from fancyimpute import IterativeImputer

def get_imputeResult_MICE(data):
    """
    Use MICE for imputation and get imputed data.

    :param data: a dictionary that contains train, test, and validation datasets
    :return: a data frame that is imputed by MICE
    """
    df = np.vstack([data['train_data'], data['val_data'], data['test_data']])
    mice_imputer = IterativeImputer()
    df_complete = mice_imputer.fit_transform(df)

    return  pd.DataFrame(df_complete)

## get the imputed data + corresponding lable for the 2nd stage task: prediction

train = pd.read_csv(path_data_raw+'train_data_with_label.cvs')
val = pd.read_csv(path_data_raw+'val_data_with_label.cvs')
test = pd.read_csv(path_data_raw+'test_data_with_label.cvs')
label = pd.concat([train['TenYearCHD'], val['TenYearCHD'], test['TenYearCHD']], axis=0)

for f_name in os.listdir(path_data_missing):

    # grab the missing data for imputation
    with open(os.path.join(path_data_missing, f_name), 'rb') as f:
        data = pickle.load(f)
    df_imputed = get_imputeResult_MICE(data)

    # output imputed data
    with open(os.path.join(path_data_imputed, 'output_data_mice_'+f_name[:12] + '.pkl'), 'wb') as f:
        pickle.dump(df_imputed, f)
