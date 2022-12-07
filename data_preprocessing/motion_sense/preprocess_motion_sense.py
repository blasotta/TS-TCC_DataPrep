import numpy as np
import pandas as pd
import torch
import os
from motionSense import set_data_types, creat_time_series, make_views, get_windows
from dataframeSplitter import DataFrameSplitter

output_dir = '../../data/MotionSense'

ACT_LABELS = ["dws","ups", "wlk", "jog", "std", "sit"]
TRIAL_CODES = {
    ACT_LABELS[0]:[1,2,11],
    ACT_LABELS[1]:[3,4,12],
    ACT_LABELS[2]:[7,8,15],
    ACT_LABELS[3]:[9,16],
    ACT_LABELS[4]:[6,14],
    ACT_LABELS[5]:[5,13]
}

# select sensor data types, typically all are wanted so set them all
# attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)
sdt = ["attitude", "gravity", "rotationRate", "userAcceleration"]
# sdt = ["userAcceleration"]
print("[INFO] -- Selected sensor data types: "+str(sdt))    
act_labels = ACT_LABELS [0:6]
print("[INFO] -- Selected activites: "+str(act_labels))    
trial_codes = [TRIAL_CODES[act] for act in act_labels]
dt_list = set_data_types(sdt)
dataset = creat_time_series(dt_list, act_labels, trial_codes, mode="raw", labeled=True)
print("[INFO] -- Shape of time-Series dataset:"+str(dataset.shape))    


print("[INFO] -- Splitting into train, val, test") 
dfs = DataFrameSplitter(method="subject")
subject_col = "id"
# These are the subjects for the train and validation set, the subjects not in
# this list will be used to build the test set, e.g. in this case
# subject ids 19-23 are in the test set (Subject-Independent split)
split_subjects = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                  9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                  17.0, 18.0]

# These are the subjects for the training set, from the split set above it
# follows that subjects 0-13 are used for train, 14-18 for validation and 
# 19-23 for testing
train_subjects = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                  9.0, 10.0, 11.0, 12.0, 13.0]

split_data, test_data = dfs.train_test_split(dataset = dataset,
                                              labels = ("id",), 
                                              subject_col=subject_col, 
                                              train_subjects=split_subjects,
                                              verbose=0)

# Split the remaining data into train and validation set, according to the trials.
# Trials 1-9 are used for training while 11-15 are used for validation, this ensures that
# each action is present in validation and training set
dfs2 = DataFrameSplitter(method="subject")
train_data, val_data = dfs2.train_test_split(dataset = split_data,
                                              labels = ("id",), 
                                              subject_col=subject_col, 
                                              train_subjects=train_subjects,
                                              verbose=0)

print("[INFO] -- Segmenting data into windows")

"""
Creation of windowed data for MotionSense:
Segment all data into windows, of 128 samples per window (at 50 Hz 2.56 s).
Step size is also 128, which indicates no overlap between windows
The function get_windows takes care of segmenting the data into windows and
returns the raw data in dimesnionality num_windows x channels x win_size, 
the corresponding labels 

For the additional information the order is as follows:

subject_id: index of the subject ranging from 0-23, for subjects 1-24
weight: weight of the subject
height: height of the subject
age: age of the subject
gender: gender of the subject, 0 encodes female, 1 encodes male
trial_id: trial that this sample came from, note that this indicates the class as in each trial only one activity was
        performed, so this may need to be removed
        
the encoding of the class labels is as follows:
"dws": 0, "ups": 1, "wlk": 2, "jog": 3, "std": 4, "sit": 5
"""

train_data, train_labels, train_info = get_windows(train_data, 128, 128)
print('Training data shape:', train_data.shape)
print('Training labels shape:', train_labels.shape)
print('Training subject info shape:', train_info.shape)
val_data, val_labels, val_info = get_windows(val_data, 128, 128)
print('Validation data shape:', val_data.shape)
print('Validation labels shape:', val_labels.shape)
print('Validation subject info shape:', val_info.shape)
test_data, test_labels, test_info = get_windows(test_data, 128, 128)
print('Test data shape:', test_data.shape)
print('Test labels shape:', test_labels.shape)
print('Test subject info shape:', test_info.shape)


print("[INFO] -- Saving Datasets") 

dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(train_data)
dat_dict["labels"] = torch.from_numpy(train_labels)
dat_dict["info"] = torch.from_numpy(train_info)
torch.save(dat_dict, os.path.join(output_dir, "train.pt"))

dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(val_data)
dat_dict["labels"] = torch.from_numpy(val_labels)
dat_dict["info"] = torch.from_numpy(val_info)
torch.save(dat_dict, os.path.join(output_dir, "val.pt"))

dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(test_data)
dat_dict["labels"] = torch.from_numpy(test_labels)
dat_dict["info"] = torch.from_numpy(test_info)
torch.save(dat_dict, os.path.join(output_dir, "test.pt"))