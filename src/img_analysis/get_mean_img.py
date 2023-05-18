import os
import skimage.io
import skimage.transform
import numpy as np

import matplotlib.pyplot as plt

PATH = "F:\Studia\INGHackathon\HackING23\src\data\datasets\\train_set_resize"

PATTERN = "\*"
SIZE = (400, 300)
directory_list = os.listdir(PATH)

for i, class_dir in enumerate(directory_list):
# for i, class_dir in enumerate(['scientific_publication', 'scientific_report', 'specification', 'umowa_na_odleglosc_odstapienie', 'umowa_o_dzielo', 'umowa_sprzedazy_samochodu']):
    _path = os.path.join(PATH, class_dir)

    img_collection = skimage.io.imread_collection(_path + PATTERN)
    num_img = len(img_collection)
    img_array = np.zeros((num_img, *SIZE), dtype=np.uint16)

    for i, img_path in enumerate(img_collection.files):
        print(class_dir, i, "/", num_img)
        img_array[i, :, :] = skimage.io.imread(img_path)
    img_mean_output = np.mean(img_array, axis=0)
    np.save(class_dir + "_mean.npy", img_mean_output)
    img_std_output = np.std(img_array, axis=0)
    np.save(class_dir + "_std.npy", img_mean_output)
    skimage.io.imsave(class_dir + "_mean.npy", img_mean_output)
    skimage.io.imsave(class_dir + "_std.npy", img_std_output)

    plt.figure()
    plt.hist(img_array.ravel(), bins=64)
    plt.savefig(class_dir + "_hist.png")
    plt.close()
