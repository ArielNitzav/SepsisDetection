import pandas as pd
import numpy as np
import math
import pickle
import sys
import os
from xgboost import XGBClassifier

with open('EDA_pickles/col_lists_dict.pkl', 'rb') as file:
    col_lists_dict = pickle.load(file)
    
with open('EDA_pickles/global_values_dict.pkl', 'rb') as file:
    global_values_dict = pickle.load(file)

def table_to_feature_vec(df):
    # create missing flags for features with missing values
    for col in col_lists_dict['cols_with_missing']:
        df[col + '_miss_flag'] = df[col].notnull().astype(int)
    
    # log transform to non-binary columns
    for col in col_lists_dict['cols_with_missing_not_binary']:
        if col in ['BaseExcess', 'O2Sat', 'FiO2', 'SaO2', 'Hct']:
            continue
        df[col] = df[col].apply(lambda x: math.log(abs(x)+1))

    # linear interpolation
    df = df.interpolate(limit_direction='both')
    
    # fill the left NAN values with the global medians found in the EDA
    for col in col_lists_dict['cols_with_missing']:
        df[col] = df[col].fillna(global_values_dict[col])
    
    # Repeat rows for timewise examination
    len_df = len(df)-1
    cols_to_repeat = col_lists_dict['cols_with_missing'] + [col + '_miss_flag' for col in col_lists_dict['cols_with_missing']] + ['ICULOS']
    sampled_rows = df[cols_to_repeat].iloc[[len_df*frac for frac in [0, 0.25, 0.5, 0.75, 1]]].fillna(0)
    finished_row = np.hstack(sampled_rows.to_numpy())
    finished_row = np.hstack([df[['Age', 'Gender', 'HospAdmTime']].iloc[0].fillna(0).to_numpy(), finished_row, df[cols_to_repeat].mean().fillna(0).to_numpy()])
    
    return finished_row

# get folder path
dir_path = sys.argv[1]

# transform the .psv files to vectors
rows = []
file_names = []
for file in os.listdir(dir_path):
    # load original df
    if file[-3:] == 'psv':
        if dir_path[-1] == '/':
            df = pd.read_csv(dir_path + file, sep='|')
        else:
            df = pd.read_csv(dir_path + '/' + file, sep='|')
        # find first row with SepsisLabel == 1
        for idx, row in df.iterrows():
            if row['SepsisLabel'] == 1:
                break
        # trim the df accordingly
        df = df.iloc[:idx+1, :]
        row = table_to_feature_vec(df)
        rows.append(row)
        file_names.append(''.join([c for c in file if c.isdigit()]))

rows = np.array(rows)
X = rows

# load the model
with open('Model_pickles/best_model_XGB.pkl', 'rb') as file:
    model = pickle.load(file)

# predict and save to a file
y_pred = model.predict(X)
results_df = pd.DataFrame({'files': file_names, 'preds': y_pred})
results_df.to_csv('prediction.csv', index=False, header=False)
