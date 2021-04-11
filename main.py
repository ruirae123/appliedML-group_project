import argparse
import pickle
import numpy as np

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor


categorical_names = [
    'male', 'education', 'currentSmoker',
    'cigsPerDay', 'BPMeds',
    'prevalentStroke', 'prevalentHyp', 'diabetes',
]

def _commandline_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_filepath', type=str, default='data/processed/framingham.csv.pkl')
  parser.add_argument('--seed', type=int, default=199)
  parser.add_argument('--miss_rate', type=float, default=0.2)
  parser.add_argument('--model_name', type=str, default='bayesian', choices=['knn', 'randomforest', 'bayesian', 'decisiontree'])

  return parser

def mask_data(X, miss_rate, random_seed):
  np.random.seed(random_seed)
  mask_test = np.random.uniform(size=X.shape) < miss_rate
  Xhat = X.copy()
  Xhat[mask_test] = np.nan

  return Xhat

def prepare_estimator(args):
  if args.model_name == 'knn':
    estimator = KNeighborsRegressor(n_neighbors=5)
  elif args.model_name == 'decisiontree':
    estimator = DecisionTreeRegressor(max_features='sqrt', random_state=0)
  elif args.model_name == 'randomforest':
    estimator = ExtraTreesRegressor(n_estimators=10, random_state=0)
  elif args.model_name == 'bayesian':
    estimator = BayesianRidge()

  return estimator

def _evalMSE(labels, preds):
  error = labels - preds
  return (error ** 2).sum(axis=-1).mean()

def _evalAcc(labels, preds):
  match = labels == preds
  return match.mean(axis=0).mean()

def main(args):
  # Preparing data
  with open(args.input_filepath, 'rb') as fd:
    data = pickle.load(fd)

  column_names = data['column_names']
  X_train = data['train_data']
  X_val = data['val_data']
  X_test = data['test_data']

  Xhat_train = mask_data(X_train, args.miss_rate, args.seed)
  # Imputation Model
  est = prepare_estimator(args)
  imp = IterativeImputer(max_iter=100, random_state=0, estimator=est)
  imp.fit(Xhat_train)

  Xhat_test = mask_data(X_test, args.miss_rate, args.seed+1)
  Pred_test = imp.transform(Xhat_test)

  print('Global Mean Squared Error: {}'.format(_evalMSE(X_test, Pred_test)))

  # Categorical Attributes
  cat_ids = [i for i, name in enumerate(column_names) if name in categorical_names ]
  Pred_cat_test = np.concatenate([Pred_test[:, i:i+1] for i in cat_ids], axis=-1)
  X_cat_test = np.concatenate([X_test[:, i:i+1] for i in cat_ids], axis=-1)
  print('Categorical Acc: {}'.format(_evalAcc(X_cat_test, Pred_cat_test)))

  # Numerical Attributes
  noncat_ids = [i for i, name in enumerate(column_names) if name not in categorical_names ]
  Pred_noncat_test = np.concatenate([Pred_test[:, i:i+1] for i in noncat_ids], axis=-1)
  X_noncat_test = np.concatenate([X_test[:, i:i+1] for i in noncat_ids], axis=-1)
  print('Numeric Mean Squared Error: {}'.format(_evalMSE(X_noncat_test, Pred_noncat_test)))

if __name__ == '__main__':
  parser = _commandline_parser()
  args = parser.parse_args()

  import warnings
  with warnings.catch_warnings():
    main(args)

