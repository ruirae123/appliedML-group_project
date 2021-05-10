# setting
from settings import *
# data preprocessing
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# modeling
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score, classification_report

def manual_model(X, y, data_version):

    # data preprocessing
    ## oversampling
    oversample = SMOTE(sampling_strategy=0.5, random_state=seed_num)
    X, y = oversample.fit_resample(X, y)

    ## test train split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed_num)

    ## scale the data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # modeling - gradient boosting
    gbc =  GradientBoostingClassifier()
    gbc.fit(X_train,y_train)
    gbc_predicted = gbc.predict(X_test)
    gbc_f1_score = f1_score(y_test, gbc_predicted)

    # report the result
    df_result = pd.DataFrame({'data_name' : [data_version],
                              'procedure_name': ['manual-GradientBoosting-default'],
                              'f1_score': [gbc_f1_score]})
    return df_result


# get the prediction result
df_result = pd.DataFrame()

for f_name in os.listdir(path_data_prediction):
    print(f_name)
    with open(os.path.join(path_data_prediction, f_name), 'rb') as f:
        df = pickle.load(f)

    df = df.sample(frac=1).reset_index(drop=True)
    X = df.iloc[:, :15]
    y = df[['TenYearCHD']]

    df_temp = manual_model(X, y, f_name)
    df_result = pd.concat([df_result, df_temp], axis=0)

df_result.to_csv(os.path.join(os.getcwd(), 'gb-result.csv'), index=False)