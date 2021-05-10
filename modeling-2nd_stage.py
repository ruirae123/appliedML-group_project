# setting
from settings import *
# data preprocessing
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# modeling
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score, classification_report
# hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV

## TODO: change the input df to try different versions of imputed data
df = df_nona.copy()

# data preprocessing
## oversampling
y = df['TenYearCHD']
X = df.drop(['TenYearCHD'], axis=1)
oversample = SMOTE(sampling_strategy=0.5, random_state=seed_num)
X, y = oversample.fit_resample(X, y)

## test train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed_num)

## scale the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# modeling - logistic regression
from sklearn.linear_model import LogisticRegression
m1 = 'LogisticRegression'
lr = LogisticRegression(random_state=seed_num, max_iter=1000)
model = lr.fit(X_train, y_train)
lr_predict = lr.predict(X_test)
lr_conf_matrix = confusion_matrix(y_test, lr_predict)
lr_f1_score = f1_score(y_test, lr_predict)
print("confusion matrix")
print(lr_conf_matrix)
print("\n")
print("f1 of Logistic Regression:",lr_f1_score,'\n')
print(classification_report(y_test,lr_predict))

# modeling - gradient boosting
m2 = 'Gradient Boosting Classifier'
gbc =  GradientBoostingClassifier()
gbc.fit(X_train,y_train)
gbc_predicted = gbc.predict(X_test)
gbc_conf_matrix = confusion_matrix(y_test, gbc_predicted)
gbc_f1_score = f1_score(y_test, gbc_predicted)
print("confussion matrix")
print(gbc_conf_matrix)
print("\n")
print("f1 score of Gradient Boosting Classifier:", gbc_f1_score)
print(classification_report(y_test,gbc_predicted))

#TODO: didn't help much here
# gradient boosting - grid search
# number of trees
n_estimators = [int(i) for i in np.linspace(start=100,stop=1500,num=100)]
# number of features to consider at every split
max_features = ['auto','sqrt']
# maximum number of levels in tree
max_depth = [int(i) for i in np.linspace(1, 10, num=1)]
# learning rate
learning_rate=[0.005, 0.01, 0.1, 0.15, 0.2]

# create the grid
gbc_grid = {'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'learning_rate': learning_rate}
# cv 5 folds
gbc=GradientBoostingClassifier(random_state=seed_num)
gbc_randomGrid = RandomizedSearchCV(estimator=gbc, param_distributions=gbc_grid,
                                    n_iter=50,
                                    scoring='f1', cv=5,
                                    verbose=0, random_state=seed_num, n_jobs=-1)
# fit the random search model
gbc_randomGrid.fit(X_train, y_train)
gbc_hyper = gbc_randomGrid.best_estimator_
gbc_hyper.fit(X_train,y_train)
gbc_predicted = gbc_hyper.predict(X_test)
gbc_f1_score = f1_score(y_test, gbc_predicted)
print("f1 of hyper-tuned Gradient Boosting Classifier:",gbc_f1_score,'\n')
print(classification_report(y_test, gbc_predicted))