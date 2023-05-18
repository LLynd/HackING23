import os
import skimage.io
import skimage.transform
import numpy as np
from skimage.filters import window
from scipy.fft import fft2, fftshift
from skimage.util import img_as_uint

PATH = "F:\Studia\INGHackathon\HackING23\src\data\datasets\\train_set_resize"
GENERAL_PATH = "F:\Studia\INGHackathon\HackING23\src\data\datasets"
PATTERN = "\*.png"
directory_list = os.listdir(PATH)

for i, class_dir in enumerate(directory_list):
    _path = os.path.join(PATH, class_dir)
    img_collection = skimage.io.imread_collection(_path + PATTERN)


    out_path_f = os.path.join(GENERAL_PATH, "ffts", class_dir)
    out_path_wf = os.path.join(GENERAL_PATH, "wffts", class_dir)
    if not os.path.exists(out_path_f):
        os.mkdir(out_path_f)

    if not os.path.exists(out_path_wf):
        os.mkdir(out_path_wf)

    for i, img_path in enumerate(img_collection.files):
        fname = img_path.rsplit('\\', 1)[-1]
        print(fname)
        image = skimage.io.imread(img_path)

        wimage = image * window('hann', image.shape)

        image_f = np.abs(fftshift(fft2(image)))
        image_fn = (image_f - np.min(image_f)) / (np.max(image_f) - np.min(image_f))
        image_fn = img_as_uint(image_fn)
        wimage_f = np.abs(fftshift(fft2(wimage)))
        wimage_fn = (wimage_f - np.min(wimage_f)) / (np.max(wimage_f) - np.min(wimage_f))


        _path_f = os.path.join(out_path_f, fname)

        skimage.io.imsave(_path_f, image_fn)
        np.save(_path_f.rsplit('.')[0] + ".npy", image_fn)

        _path_wf = os.path.join(out_path_wf, fname)
        skimage.io.imsave(_path_wf, wimage_fn)
        np.save(_path_wf.rsplit('.')[0] + ".npy", wimage_fn)


