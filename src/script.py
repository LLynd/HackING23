import os

import pandas as pd
import typing as t
import numpy as np

from PIL import Image
from sklearn.model_selection import train_test_split

IMG_TEST_PATH = os.path.join("data", "datasets", "test_set")

i = 0
for img in os.listdir(IMG_TEST_PATH):
    print(img)
    if i == 0:
        shape = list(np.array(Image.open(os.path.join(IMG_TEST_PATH, img))).shape)
        shape.insert(0, len(os.listdir(IMG_TEST_PATH)))
        print(shape)
        X =  np.zeros(tuple(shape), dtype=np.float32)
    X[i] = np.array(Image.open(os.path.join(IMG_TEST_PATH, img)).resize(shape[1:], Image.Resampling.LANCZOS))
    i+=1
print(X.shape)