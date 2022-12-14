import torch
import os
from unimib_helpers import create_data

output_dir = '../../data/UnimibShar'

'''
Select which subjects 1-30 should be in the train, validation and test splits
in the three lists below
'''

trn_subs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 ,17, 18, 19,
            20, 21, 22]
val_subs =[23, 24, 25, 26]
tst_subs =[27, 28, 29, 30]

'''
Choose the activity type for which acceleration data should be loaded by setting the variable act.
There are 4 options:
    - 'acc' loads accleration data for all 17 activities with seperate label each; labels 0-16
    - 'adl' only loads accleration data for the 9 Activities of daily living labels; 0-8
    - 'fall' only loads accleration data for the 8 types of falls; labels 0-7
    - 'two_classes' loads accleration data of all activities with only two class labels 0=adl, 1=fall
    
The classes 0-16 are the following:
    ['Standing up from sitting', 'Standing up from laying', 'Walking', 'Running', 'Going upstairs',
     'Jumping', 'Going downstairs', 'Lying down from standing', 'Sitting down',
     'Generic falling forward', 'Falling rightward', 'Generic falling backward',
     'Hitting an obstacle in the fall', 'Falling with protection strategies',
     'Falling backward-sitting-chair', 'Syncope', 'Falling leftward']
    
The data that is loaded is windowed data with windowsize of 151 around peaks of 1.5G, consiting
of three channels (accelerometer data on the x,y,z axis). Shape = (N x 3 x 151)

For the additional information the order is as follows:
    subject_id: index of the subject ranging from 1-30
    weight: weight of the subject
    height: height of the subject
    age: age of the subject
    gender: gender of the subject, 0 encodes female, 1 encodes male
    trial_id: trial that this sample came from
'''
act = 'acc' # options 'acc', 'adl', 'fall', 'two_classes'

trn_dict, val_dict, tst_dict = create_data(trn_subs, val_subs, tst_subs, act)



torch.save(trn_dict, os.path.join(output_dir, "train.pt"))
torch.save(val_dict, os.path.join(output_dir, "val.pt"))
torch.save(tst_dict, os.path.join(output_dir, "test.pt"))