import torch
import os
from carrots_helpers import get_carrots

output_dir = '../../data/Carrots'

'''
Select which subjects 1-7 should be in the train, validation and test splits
in the three lists below.

Also set the window size and the step size in terms of samples, step_size = win_size indicates no overlap
'''

trn_subs = [1, 2, 3, 4, 5]
val_subs =[6]
tst_subs =[7]
win_size = 128
step_size = 128

print("[INFO] -- Loading Carrots Data")

X_trn, y_trn, i_trn, X_val, y_val, i_val, X_tst, y_tst, i_tst = get_carrots(win_size, step_size, trn_subs, val_subs, tst_subs)

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