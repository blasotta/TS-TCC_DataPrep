from scipy.io import arff
from scipy import stats
from sklearn import preprocessing
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import as_strided
import os

DATAPATH = 'D2011-KTA-KHY/001-IMU'

def load_dataset(number):
    dataset, meta = arff.loadarff(os.path.join(DATAPATH, f'raw_{number}.arff'))
    labels = dataset['class']
    df = pd.DataFrame(dataset)
    df.drop(columns=['time', 'class'], inplace=True)

    for i in range(len(labels)):
        s = labels[i].decode('UTF-8')
        split = s.split('-', 1)
        labels[i] = split[0]

    str_labels = labels.astype('str')
    le = preprocessing.LabelEncoder()
    le.fit(str_labels)
    y = le.transform(str_labels)
    X = df.to_numpy()
    return X, y


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


def get_windows(X,y, win_size, step_size):
    X = np.ascontiguousarray(X)
    views = make_views(X, win_size, step_size)
    X = np.transpose(views, (0,2,1))
        
    y = y.astype('int64') # change labels to int64 for windowing correctly
    y = np.ascontiguousarray(y)
    y_2d = np.reshape(y, (y.shape[0],1))
    y_views = make_views(y_2d, win_size, step_size)
    y_newshape = (y_views.shape[0], y_views.shape[1]*y_views.shape[2])
    y_new = np.reshape(y_views, newshape=y_newshape)
    y,_ = stats.mode(y_new, axis=1, keepdims=False)
    
    assert (y.shape[0] == X.shape[0])
    
    return X,y

def get_sub_data(subs, win_size, step_size):
    data = []
    labels = []
    info = []
    for sub in subs:
        X, y = load_dataset(sub)
        X, y = get_windows(X, y, win_size, step_size)
        sinf = np.full(X.shape[0], sub)
        data.append(X)
        labels.append(y)
        info.append(sinf)
        
    X = np.concatenate(data)
    y = np.concatenate(labels)
    i = np.concatenate(info)
    
    return X, y, i


def get_carrots(win_size, step_size, trn_subs, val_subs, tst_subs):
    X_trn, y_trn, i_trn = get_sub_data(trn_subs, win_size, step_size)
    X_val, y_val, i_val = get_sub_data(val_subs, win_size, win_size)
    X_tst, y_tst, i_tst = get_sub_data(tst_subs, win_size, win_size)
    # step size set to win_size because no overlap in validation or test data wanted
    
    
    return X_trn, y_trn, i_trn, X_val, y_val, i_val, X_tst, y_tst, i_tst