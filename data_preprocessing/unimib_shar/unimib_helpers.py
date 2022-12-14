import numpy as np
import pandas as pd
import torch
import os
from pymatreader import read_mat

pd.set_option('display.max_columns', None)

DATAPATH = 'UniMiB-SHAR/data'

def load_mat(fname):
    data = read_mat(os.path.join(DATAPATH, fname))
    return data


def get_ds_infos():
    """
    Read the file with data subject information.
    
    Data Columns:
    0: id [1-30]
    1: gender  [0:Female, 1:Male]
    2: age [years] 
    3: height [cm]
    4: weight [kg]
    
    Don't worry about the order of columns, they will be reordered later to match
    the order of additional info of the MotionSense dataset
    
    Returns:
        A pandas DataFrame that contains inforamtion about data subjects' attributes 
    """ 

    dss = pd.read_csv("data_subjects_info.csv")
    
    return dss

def load_unimib(act='acc'):
    xpath = act + '_data.mat'
    ypath = act + '_labels.mat'
    ipath = act + '_names.mat'
    data = read_mat(os.path.join(DATAPATH, xpath))[f'{act}_data']
    data = np.reshape(data, (-1, 3, 151))
    labels = read_mat(os.path.join(DATAPATH, ypath))[f'{act}_labels']
    info = read_mat(os.path.join(DATAPATH, ipath))[f'{act}_names']
    return data, labels, info

def get_split(data, label, sinfo, sub):
    x = []
    i = []
    y = []
    for s in sub:
        idx = (sinfo[:,0] == s).nonzero()
        inf = sinfo[idx]
        spy = label[idx]
        dat = data[idx]
        
        x.append(dat)
        i.append(inf)
        y.append(spy)
        
    data = np.concatenate(x)
    labels = np.concatenate(y)
    info = np.concatenate(i)
    
    return data, labels, info

def make_dict(data, labels, info):
    dat_dict = dict()
    dat_dict["samples"] = torch.from_numpy(data)
    dat_dict["labels"] = torch.from_numpy(labels)
    dat_dict["info"] = torch.from_numpy(info)
    
    return dat_dict

def create_data(trn_subs, val_subs, tst_subs, act):
    data, labels, info = load_unimib(act)

    dfl = pd.DataFrame(labels, columns=['class', 'id', 'trial'])
    dss = get_ds_infos()

    df = dfl.join(dss.set_index('id'), on='id')
    df = df.reindex(columns=['class', 'id', ' weight', ' height', ' age', ' gender', 'trial'])

    subject_info = df.to_numpy()

    split = np.hsplit(subject_info, [1])
    label = split[0].flatten()
    label = label - np.min(label)
    sinfo = split[1]

    trn_data, trn_labels, trn_info = get_split(data, label, sinfo, trn_subs)
    print("[INFO] -- Shape of train set:", trn_data.shape)
    print("[INFO] -- Shape of train subject info:", trn_info.shape)
    val_data, val_labels, val_info = get_split(data, label, sinfo, val_subs)
    print("[INFO] -- Shape of val set:", val_data.shape)
    print("[INFO] -- Shape of train subject info:", val_info.shape)
    tst_data, tst_labels, tst_info = get_split(data, label, sinfo, tst_subs)
    print("[INFO] -- Shape of test set:", tst_data.shape)
    print("[INFO] -- Shape of train subject info:", tst_info.shape)
    
    trn_dict = make_dict(trn_data, trn_labels, trn_info) 
    val_dict = make_dict(val_data, val_labels, val_info)
    tst_dict = make_dict(tst_data, tst_labels, tst_info)
    
    return trn_dict, val_dict, tst_dict