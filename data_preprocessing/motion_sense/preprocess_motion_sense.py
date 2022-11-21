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
# These are the subjects for the train set 0-23 the subjects not in this list
# will be used to build the test set, e.g. in this case subject 24 (index 23)
# is the test subject (Subject-Independent split)
train_subjects = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                  9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                  17.0, 18.0, 19.0, 20.0, 21.0, 22.0]

split_data, test_data = dfs.train_test_split(dataset = dataset,
                                              labels = ("id",), 
                                              subject_col=subject_col, 
                                              train_subjects=train_subjects,
                                              verbose=0)

# Split the remaining data into train and validation set, according to the trials.
# Trials 1-9 are used for training while 11-15 are used for validation, this ensures that
# each action is present in validation and training set
dfs2 = DataFrameSplitter(method="trials")
train_data, val_data = dfs2.train_test_split(dataset = split_data,
                                              labels = ("id","trial"), 
                                              trial_col='trial', 
                                              train_trials=[1.,2.,3.,4.,5.,6.,7.,8.,9.],
                                              verbose=0)

print("[INFO] -- Segmenting data into windows") 
# Segment all data into windows, of 5 seconds at 50 Hz (= 250 samples make one window).
# For training data use overlap of 50% for the rest no overlap
train = get_windows(train_data, 250, 125)
val = get_windows(val_data, 250, 250)
test = get_windows(test_data, 250, 250)

"""
After loading the data it is split into the raw data, the label and the additional
information about the subject. The raw data is simply a concatenation of the window, i.e.,
each window is a 250 x 12 array, which is reshaped to a 3000-d feature vector

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

train_split = np.hsplit(train, [3000, 3001])
val_split = np.hsplit(val, [3000, 3001])
test_split = np.hsplit(test, [3000, 3001])

X_train = train_split[0]
X_val = val_split[0]
X_test = test_split[0]

y_train = train_split[1].flatten()
y_val = val_split[1].flatten()
y_test = test_split[1].flatten()

info_train = train_split[2]
info_val = val_split[2]
info_test = test_split[2]

print("[INFO] -- Saving Datasets") 

dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(X_train)
dat_dict["labels"] = torch.from_numpy(y_train)
dat_dict["info"] = torch.from_numpy(info_train)
torch.save(dat_dict, os.path.join(output_dir, "train.pt"))

dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(X_val)
dat_dict["labels"] = torch.from_numpy(y_val)
dat_dict["info"] = torch.from_numpy(info_val)
torch.save(dat_dict, os.path.join(output_dir, "val.pt"))

dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(X_test)
dat_dict["labels"] = torch.from_numpy(y_test)
dat_dict["info"] = torch.from_numpy(info_test)
torch.save(dat_dict, os.path.join(output_dir, "test.pt"))