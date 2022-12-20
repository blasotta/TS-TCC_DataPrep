import zipfile
import numpy as np
import pandas as pd
import re
import os
from pathlib import Path
from numpy.lib.stride_tricks import as_strided
    
    
def get_subject_infos():
    """
    Read the file includes data subject information.
    
    Data Columns:
    0: code [1-15]
    1: weight [kg]
    2: height [cm]
    3: age [years]
    4: gender [0:Female, 1:Male]
    
    Returns:
        A pandas DataFrame that contains inforamtion about data subjects' attributes 
    """ 

    dss = pd.read_csv("data_subjects_info.csv")
    
    return dss

def make_views(arr, win_size, step_size, writeable = False):
  """
  arr: input 2d array to be windowed
  win_size: size of data window (given in data points)
  step_size: size of window step (given in data point)
  writable: if True, elements can be modified in new data structure, which will affect
    original array (defaults to False)
  Note that data should be of type 64 bit (8 byte)
  
  Returns: view on array of shape (num_windows, win_size, n_columns)
  """
  
  n_records = arr.shape[0]
  n_columns = arr.shape[1]
  remainder = (n_records - win_size) % step_size 
  num_windows = 1 + int((n_records - win_size - remainder) / step_size)
  new_view_structure = as_strided(
    arr,
    shape = (num_windows, win_size, n_columns),
    strides = (8 * step_size * n_columns, 8 * n_columns, 8),
    writeable = False,
  )
  return new_view_structure


def get_windows(X, win_size, step_size):
    X = np.ascontiguousarray(X)
    views = make_views(X, win_size, step_size)
    X = np.transpose(views, (0,2,1))
    
    return X

def extract_data(in_path, out_path):
    with zipfile.ZipFile(in_path, 'r') as zip_ref:
        zip_ref.extractall(out_path)
        

def extract_nested_zip(zippedFile, toFolder):
    try:
        with zipfile.ZipFile(zippedFile, 'r') as zfile:
            zfile.extractall(path=toFolder)
    except: 
        pass
    
    try:
        os.remove(zippedFile)
    except:
        pass
    for root, dirs, files in os.walk(toFolder):
        for filename in files:
            if re.search(r'\.zip$', filename):
                fileSpec = os.path.join(root, filename)
                extract_nested_zip(fileSpec, root)

def extract_all(act_list, loc_list):
    for i in range(1,16):
        for act in act_list:
            Path(f'proband{i}/extracted_data').mkdir(parents=True, exist_ok=True)
            out_path = f'proband{i}/extracted_data'
            extract_nested_zip(f'proband{i}/data/acc_{act}_csv.zip', out_path)
            extract_nested_zip(f'proband{i}/data/gyr_{act}_csv.zip', out_path)
            
def load_data(sub, act, sens, loc):
    fname = f'proband{sub}/extracted_data/{sens}_{act}_{loc}.csv'
    if ((sub == 4) & (act == 'walking')) or ((sub == 6) & (act == 'sitting')) or \
       ((sub == 7) & (act == 'sitting')) or ((sub == 8) & (act == 'standing')) or \
       ((sub == 13) & (act == 'walking')):
        fname = f'proband{sub}/extracted_data/{sens}_{act}_2_{loc}.csv'
    df = pd.read_csv(fname, header=0,
                     names=['id', 't', f'{sens}_{loc}_x', f'{sens}_{loc}_y', f'{sens}_{loc}_z'],
                     usecols=[2,3,4])
    return df

'''
This Function loads data for the given subject and activity and stacks the
channels given as sens_list, and loc_list together. The data is the segmented
into windows of the given size. Additional subject info is also loaded.

For subject info 5 fields are available:
    0: subject_id [1-15]
    1: weight [kg]
    2: height [cm]
    3: age [years]
    4: gender [0:Female, 1:Male]

Returns: windowed data, label per window and subject info per window as numpy arrays
'''
def stack_data(sub, label, act, sens_list, loc_list, win_size, step_size):
    frames = []
    for sens in sens_list:
        for loc in loc_list:
            df = load_data(sub, act, sens, loc)
            frames.append(df)
    
    df = pd.concat(frames, axis=1)
    df.dropna(inplace=True)
    data = df.to_numpy()
    
    X = get_windows(data, win_size, step_size)
    
    dss = get_subject_infos()
    sinf = dss.iloc[sub-1]
    
    info = np.array([label, sub, sinf[1], sinf[2], sinf[3], sinf[4]])
    subject_info = np.tile(info, (X.shape[0], 1))
    
    split = np.hsplit(subject_info, [1])
    y = split[0].flatten()
    i = split[1]

    return X, y, i

def get_split_data(trn_subs, val_subs, tst_subs, act_list, sens_list, loc_list, win_size, step_size):
    X_trn = []
    y_trn = []
    i_trn = []
    
    X_val = []
    y_val = []
    i_val = []
    
    X_tst = []
    y_tst = []
    i_tst = []
    for sub in trn_subs:
        for label, act in enumerate(act_list):
            if (sub==6) & (act=='jumping'):
                continue
            else:
                X, y, i = stack_data(sub, label, act, sens_list, loc_list, win_size, step_size)
                X_trn.append(X)
                y_trn.append(y)
                i_trn.append(i)
            
    for sub in val_subs:
        for label, act in enumerate(act_list):
            if (sub==6) & (act=='jumping'):
                continue
            else:
                X, y, i = stack_data(sub, label, act, sens_list, loc_list, win_size, win_size)
                X_val.append(X)
                y_val.append(y)
                i_val.append(i)
            
    for sub in tst_subs:
        for label, act in enumerate(act_list):
            if (sub==6) & (act=='jumping'):
                continue
            else:
                X, y, i = stack_data(sub, label, act, sens_list, loc_list, win_size, win_size)
                X_tst.append(X)
                y_tst.append(y)
                i_tst.append(i)
            
    X_trn = np.concatenate(X_trn)
    y_trn = np.concatenate(y_trn)
    i_trn = np.concatenate(i_trn)
    
    X_val = np.concatenate(X_val)
    y_val = np.concatenate(y_val)
    i_val = np.concatenate(i_val)
    
    X_tst = np.concatenate(X_tst)
    y_tst = np.concatenate(y_tst)
    i_tst = np.concatenate(i_tst)
    
    return X_trn, y_trn, i_trn, X_val, y_val, i_val, X_tst, y_tst, i_tst