import os
import pandas as pd
import skimage.io
import matplotlib.pyplot as plt

PATH = "F:\Studia\INGHackathon\HackING23\src\data\datasets\\train_set"
DOWNSAMPLED_TRAIN = "F:\Studia\INGHackathon\HackING23\src\data\datasets\\train_set_ds"

PATTERN = "\*"

directory_list = os.listdir(PATH)

for class_dir in directory_list:
    _path = os.path.join(PATH, class_dir)
    #files_list = os.listdir(_path)

    img_collection = skimage.io.imread_collection(_path + PATTERN)
    for img_path in img_collection.files:
        img = skimage.io.imread(img_path)





