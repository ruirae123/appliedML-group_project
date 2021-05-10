from settings import *

# get the original missing data
for f_name in os.listdir(path_data_imputed):

    output_data = dict()

    if 'bayesian' in f_name and '0.1' in f_name:

        with open(os.path.join(path_data_imputed, f_name), 'rb') as f:
            data = pickle.load(f)
            output_data['train_data'] = data['train_data']
            output_data['val_data'] = data['val_data']
            output_data['test_data'] = data['test_data']

            with open(os.path.join(path_data_missing, 'missrate=0.1.pkl'), 'wb') as f:
                pickle.dump(output_data, f)

    if 'bayesian' in f_name and '0.2' in f_name:

        with open(os.path.join(path_data_imputed, f_name), 'rb') as f:
            data = pickle.load(f)
            output_data['train_data'] = data['train_data']
            output_data['val_data'] = data['val_data']
            output_data['test_data'] = data['test_data']

            with open(os.path.join(path_data_missing, 'missrate=0.2.pkl'), 'wb') as f:
                pickle.dump(output_data, f)

    if 'bayesian' in f_name and '0.3' in f_name:

        with open(os.path.join(path_data_imputed, f_name), 'rb') as f:
            data = pickle.load(f)
            output_data['train_data'] = data['train_data']
            output_data['val_data'] = data['val_data']
            output_data['test_data'] = data['test_data']

            with open(os.path.join(path_data_missing, 'missrate=0.3.pkl'), 'wb') as f:
                pickle.dump(output_data, f)

    if 'bayesian' in f_name and '0.4' in f_name:

        with open(os.path.join(path_data_imputed, f_name), 'rb') as f:
            data = pickle.load(f)
            output_data['train_data'] = data['train_data']
            output_data['val_data'] = data['val_data']
            output_data['test_data'] = data['test_data']

            with open(os.path.join(path_data_missing, 'missrate=0.4.pkl'), 'wb') as f:
                pickle.dump(output_data, f)

    if 'bayesian' in f_name and '0.5' in f_name:

        with open(os.path.join(path_data_imputed, f_name), 'rb') as f:
            data = pickle.load(f)
            output_data['train_data'] = data['train_data']
            output_data['val_data'] = data['val_data']
            output_data['test_data'] = data['test_data']

            with open(os.path.join(path_data_missing, 'missrate=0.5.pkl'), 'wb') as f:
                pickle.dump(output_data, f)

# put together data for prediction
## get the imputed data + corresponding lable for the 2nd stage task: prediction

train = pd.read_csv(path_data_raw+'train_data_with_label.cvs')
val = pd.read_csv(path_data_raw+'val_data_with_label.cvs')
test = pd.read_csv(path_data_raw+'test_data_with_label.cvs')
label = pd.concat([train['TenYearCHD'], val['TenYearCHD'], test['TenYearCHD']], axis=0)

for f_name in os.listdir(path_data_imputed):

    if 'mice' not in f_name:

        with open(os.path.join(path_data_imputed, f_name), 'rb') as f:
            data = pickle.load(f)
        df_imputed = pd.DataFrame(np.vstack([data['imputed_train_data'],
                                             data['imputed_val_data'],
                                             data['imputed_test_data']]))
        df_imputed['TenYearCHD'] = label.tolist()

        # output imputed data
        with open(os.path.join(path_data_prediction, 'prediction_data_'+'_'.join(f_name.split('_')[2:4]) + '.pkl'), 'wb') as f:
            pickle.dump(df_imputed, f)

    else:
        with open(os.path.join(path_data_imputed, f_name), 'rb') as f:
            df_imputed = pickle.load(f)
        df_imputed['TenYearCHD'] = label.tolist()
        with open(os.path.join(path_data_prediction, 'prediction_data_'+'_'.join(f_name.split('_')[2:4])), 'wb') as f:
            pickle.dump(df_imputed, f)