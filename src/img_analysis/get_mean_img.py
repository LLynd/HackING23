import os
import skimage.io
import skimage.transform
import numpy as np

import matplotlib.pyplot as plt

PATH = "F:\Studia\INGHackathon\HackING23\src\data\datasets\\train_set_resize"


PATTERN = "\*"
SIZE = (800, 600)
directory_list = os.listdir(PATH)

for i, class_dir in enumerate(directory_list):
    _path = os.path.join(PATH, class_dir)

    img_collection = skimage.io.imread_collection(_path + PATTERN)
    img_array = np.zeros((len(img_collection), *SIZE), dtype=np.uint8)
    img_mean_output = np.zeros(SIZE, dtype=np.uint8)
    for i, img_path in enumerate(img_collection.files):
        img_array[i,:,:] = skimage.io.imread(img_path)
    img_mean_output[i] = np.mean(img_array, axis=0)
    skimage.io.imsave(class_dir + "_mean.png", img_mean_output)
    plt.figure()
    plt.hist(img_array.ravel(), bins = 256)
    plt.savefig(class_dir + "_hist.png")
    plt.close()
