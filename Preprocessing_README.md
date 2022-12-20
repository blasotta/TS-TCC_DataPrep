# Instructions for Preprocessing New Datasets

These are instructions for processing four new Datasets for TS-TCC. All data will be saved as pytorch files in the folder `data/subfolder`, with the names train.pt,
val.pt and test.pt. Each of them is a dictionary with three keys 'samples', 'labels', 'info' and under the key the corresponding data is saved as a torch tensor.
For the carrots dataset no additional subject information is available and thus the value of key 'info' is None.

## MotionSense

1. Download [MotionSense](https://github.com/mmalekzadeh/motion-sense/blob/261cb0d60611a049ab38127738e42cc510ed50f4/data/A_DeviceMotion_data.zip)
2. Unpack folder and place into directory `data_preprocessing/motion_sense`
3. In the main directory create the folder `data` and within the folder `MotionSense`. The processed .pt files will be placed under `data/MotionSense`
4. Run the script `preprocess_motion_sense.py`
5. If wanted, the selected sensor features , train-test split, selected activities and window size can be adjusted in the file `preprocess_motion_sense.py`

## UniMiB SHAR

1. Download [UniMiB-SHAR.zip](https://www.dropbox.com/s/x2fpfqj0bpf8ep6/UniMiB-SHAR.zip?dl=0)
2. Unpack and place folder UniMiB-SHAR into the directory `data_preprocessing/unimib_shar`
3. In the main directory create the folder `UnimibShar` in the folder `data`. Such that path `data/UnimibShar` exists
4. Install the package pymatreader to read matlab files into numpy arrays: For conda use `conda install -c conda-forge pymatreader`; for pip do `pip install pymatreader`
5. Run the script `preprocess_unimib.py`
6. If wanted, the train-test split and selected activities can be adjusted in the file `preprocess_unimib.py`

## Carrots Dataset

1. Download the [Carrot soup cooking dataset](https://rosdok.uni-rostock.de/resolve/id/rosdok_document_0000010639)
2. Unpack and place folder `D2011-KTA-KHY` into the directory `data_preprocessing/carrots`
3. In the main directory create the folder `Carrots` in the folder `data`. Such that path `data/Carrots` exists
4. Run the script `preprocess_carrots.py`
5. If wanted, the train-test split and window_size can be adjusted in the file `preprocess_unimib.py`

## Realworld Dataset

1. Download the [RealWorld dataset](https://www.uni-mannheim.de/dws/research/projects/activity-recognition/dataset/dataset-realworld/)
2. Unzip contents into directory `data_preprocessing/realworld`
3. In the main directory create the folder `Realworld` in the folder `data`. Such that path `data/Realworld` exists
4. Run the script `preprocess_realworld.py`
5. If wanted, the subjects for train, validation, test_set as well as window size and sensor types can be set
