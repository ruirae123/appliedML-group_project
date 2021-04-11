import os
import argparse
import constant
import pickle
import numpy as np


def _commandline_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_filepath', type=str, default='data/processed/framingham.csv.mr=0.500.pkl')

  return parser

if __name__ == '__main__':
  parser = _commandline_parser()
  args = parser.parse_args()

  with open(args.input_filepath, 'rb') as fd:
    data = pickle.load(fd)

  print('Shape of training data: {}'.format(data['train_data'].shape))
  print('Shape of validation data: {}'.format(data['test_data'].shape))
  print('Shape of test data: {}'.format(data['test_data'].shape))
