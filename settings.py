# import basic packages
import os
import pickle
import pandas as pd
import numpy as np
import re

# set up the paths
path_data_raw = "C:/Users/Rui/Desktop/MIT/Academic/0-AppliedML/appliedML-group_project/data_raw/"
path_data_imputed = "C:/Users/Rui/Desktop/MIT/Academic/0-AppliedML/appliedML-group_project/data_imputed/"
path_data_missing = "C:/Users/Rui/Desktop/MIT/Academic/0-AppliedML/appliedML-group_project/data_missing/"
path_data_prediction = "C:/Users/Rui/Desktop/MIT/Academic/0-AppliedML/appliedML-group_project/data_prediction/"
# set up the file name
f_data_raw = "framingham.csv"

# read in file
df_raw = pd.read_csv(path_data_raw+f_data_raw) # original
df_nona = df_raw[~df_raw.isnull().any(axis=1)].reset_index() # without rows that have missing values
df_nona.drop(['index'], axis=1, inplace=True)

# miscellaneous
seed_num = 123