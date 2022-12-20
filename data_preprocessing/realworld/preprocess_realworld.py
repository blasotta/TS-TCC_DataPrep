import torch
import os
from realworld_helpers import extract_all, get_split_data

output_dir = '../../data/Realworld'


trn_subs = [1,2,3,4,5,6,7,8,9,10,11]
val_subs = [12,13]
tst_subs = [14,15]

act_list = ['climbingdown', 'climbingup', 'jumping', 'lying', 'running',
            'sitting', 'standing', 'walking']
sens_list = ['acc', 'Gyroscope']
loc_list = ['chest', 'head', 'shin', 'thigh', 'upperarm', 'waist']
# removed sensor location 'forearm' as this sensor position is not available for all subjects

win_size = 128
step_size = 128


if (not os.path.exists('proband1/extracted_data/acc_climbingdown_chest.csv')):
    print("[INFO] -- Extracting ziped data")
    extract_all(act_list, loc_list)
    
print("[INFO] -- Loading and windowing data")
X_trn, y_trn, i_trn, X_val, y_val, i_val, X_tst, y_tst, i_tst = (
    get_split_data(trn_subs, val_subs, tst_subs, act_list, sens_list, loc_list, win_size, step_size)
    )

print("[INFO] -- Shape of train set:", X_trn.shape)
print("[INFO] -- Shape of validation set:", X_val.shape)
print("[INFO] -- Shape of test set:", X_tst.shape)

X_train = X_trn.copy()
y_train = y_trn.copy()
X_valid = X_val.copy()
y_valid = y_val.copy()
X_test = X_tst.copy()
y_test = y_tst.copy()

print("[INFO] -- Creating and saving Pytorch datasets")

dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(X_train)
dat_dict["labels"] = torch.from_numpy(y_train)
dat_dict["info"] = torch.from_numpy(i_trn)
torch.save(dat_dict, os.path.join(output_dir, "train.pt"))

dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(X_valid)
dat_dict["labels"] = torch.from_numpy(y_valid)
dat_dict["info"] = torch.from_numpy(i_val)
torch.save(dat_dict, os.path.join(output_dir, "val.pt"))

dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(X_test)
dat_dict["labels"] = torch.from_numpy(y_test)
dat_dict["info"] = torch.from_numpy(i_tst)
torch.save(dat_dict, os.path.join(output_dir, "test.pt"))