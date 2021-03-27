# import basic packages
import os
import pandas as pd
import numpy as np
import re

# set up the paths
path_data_raw = "C:/Users/Rui/Desktop/MIT/Academic/0-AppliedML/project/data_raw/"
path_data_processed = "C:/Users/Rui/Desktop/MIT/Academic/0-AppliedML/project/data_processed/"

# set up the file name
f_data_raw = "framingham.csv"

# read in file
df_raw = pd.read_csv(path_data_raw+f_data_raw) # original
df_nona = df_raw[~df_raw.isnull().any(axis=1)].reset_index() # without rows that have missing values

# miscellaneous
seed_num = 123