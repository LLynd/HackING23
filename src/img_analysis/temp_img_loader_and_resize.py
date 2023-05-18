import os
import numpy as np
import skimage.io
import skimage.transform
from skimage.util import img_as_uint
from tqdm import tqdm

PATH = "F:\Studia\INGHackathon\HackING23\src\data\datasets\\train_set"
DOWNSAMPLED_TRAIN = "F:\Studia\INGHackathon\HackING23\src\data\datasets\\train_set_resize"

PATTERN = "\*"

directory_list = os.listdir(PATH)

for class_dir in directory_list:
    _path = os.path.join(PATH, class_dir)
    #files_list = os.listdir(_path)
    print(_path)
    img_collection = skimage.io.imread_collection(_path + PATTERN)
    _out_path = os.path.join(DOWNSAMPLED_TRAIN, class_dir)
    if not os.path.exists(_out_path):
        os.mkdir(_out_path)

    for img_path in tqdm(img_collection.files):

        img = skimage.io.imread(img_path)
        if img.shape[1] > img.shape[0]:
            img = np.fliplr(np.transpose(img, (1, 0)))

        img_r = skimage.transform.resize(img, (400, 300))
        img_r = img_as_uint(img_r)
        _path = img_path.replace("train_set", "train_set_resize")
        _path = _path.replace(".tiff", ".png")
        _path = _path.replace(".jpg", ".png")
        skimage.io.imsave(_path, img_r)




