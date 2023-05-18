import os

import pandas as pd
import typing as t
import numpy as np

from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class DataLoader():
    IMG_TEST_PATH = os.path.join("data", "datasets", "test_set")
    IMG_TRAIN_PATH = os.path.join("data", "datasets", "train_set")
    TXT_PATH = os.path.join("data", "hackathon")
    
    def __init__(self, random_state: t.Optional[int] = 42) -> None:
        self.random_state = random_state
    
    
    def load(self, data: str):
        shape = list(np.array(Image.open(os.path.join(self.IMG_TEST_PATH, img))).shape)
    
        if data == 'IMG_TEST':
            i = 0
            shape.insert(0, len(os.listdir(self.IMG_TEST_PATH)))
            X =  np.zeros(tuple(shape), dtype=np.float32)
            for img in tqdm(os.listdir(self.IMG_TEST_PATH)):
                X[i, :, :] = np.array(Image.open(os.path.join(self.IMG_TEST_PATH, img)).resize(shape[1:], Image.Resampling.LANCZOS)).T
                i += 1
            return X
        
        elif data == "IMG_TRAIN":
            i = 0
            shape.insert(0, sum([len(files) for r, d, files in os.walk(self.IMG_TRAIN_PATH)]))
            X =  np.zeros(tuple(shape), dtype=np.float32)
            y = []
            for dir in tqdm(os.listdir(self.IMG_TRAIN_PATH)):
                for img in tqdm(os.listdir(os.path.join(self.IMG_TRAIN_PATH, dir))):
                    y.append(dir)
                    X[i, :, :] = np.array(Image.open(os.path.join(self.IMG_TEST_PATH, img)).resize(shape[1:], Image.Resampling.LANCZOS)).T
                    i += 1
            return X, np.array(y)
        
        elif data == "TXT_TEST":
            
            return X
        
        elif data == "IMG_TRAIN":
        
            return X, y
    
    
    def split(self, X: np.typing.ArrayLike, y: np.typing.ArrayLike, test_size: t.Optional[float] = 0.2, shuffle: t.Optional[bool] = True, stratify: t.Optional[np.typing.ArrayLike] = None):
        return train_test_split(X, y, test_size=test_size, shuffle=shuffle, stratify=stratify, random_state=self.random_state)
    
    
    def preprocess(self, data:str):
        pass