import json

import numpy as np
from tensorflow import keras
import tensorflow as tf
import keras.backend as K
import pandas as pd
import os
from PIL import Image

class DataGen(keras.utils.Sequence):
    def __init__(self, input_shape=(384,384,1), data_path='./data', mode='train', batch_size=32, shuffle=True, quiet=False, norm='sample'):
        print(f'initialising {mode} generator...')
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.quiet = quiet
        self.shuffle = shuffle
        self.mode = mode
        self.norm = norm
        self.text_path = os.path.join(data_path, 'hackaton')
        self.index= pd.DataFrame()




        if self.mode=='train':
            self.img_path = os.path.join(data_path, 'datasets', 'train_set')
            #labels_file = 'train_labels_final.json'
            #text_file = 'train_set_ocr.json'
            """
            with open(os.path.join(self.text_path, labels_file), 'r') as f:
                data = json.load(f)
            self.index['label'] = data.values()
            self.index['fname'] = data.keys()
            """
        elif self.mode=='test':
            self.img_path = os.path.join(data_path, 'datasets', 'test_set')
            #text_file = 'test_ocr_clean.json'
        else:
            raise ValueError('Unsupported mode')

        """
        with open(os.path.join(self.text_path, text_file), 'r') as f:
            data = json.load(f)

        if self.mode == 'train':
            temporary_df = pd.DataFrame()
            temporary_df['fname'] = data.keys()
            temporary_df['text'] = data.values()
            self.index = pd.merge(left=self.index, right=temporary_df, on='fname')

        elif mode == 'test':
            self.index['fname'] = data.keys()
            self.index['text'] = data.values()
        """

        self.index = self._make_index()

        if self.mode=='train':
            self.classes = np.asarray(self.index['label'].unique())
        else:
            self.classes = np.zeros(0)

        self.available_ids = list(self.index.index)
        self.on_epoch_end()
        print(f'Generator fully initialised, {len(self.index)} samples available')

    def _make_index(self):
        df = pd.DataFrame()
        if self.mode == 'train':
            for directory in os.listdir(self.img_path):
                dir_df = pd.DataFrame()
                if directory[0] != '.':
                    dirfiles = os.listdir(os.path.join(self.img_path, directory))
                    dir_df['fname'] = dirfiles
                    dir_df['label'] = [directory]*len(dir_df)
                    df = pd.concat([df, dir_df], ignore_index=True)
        else:
            files = os.listdir(self.img_path)
            df['fname'] = files
        return df

    def __len__(self):
        return int(np.floor(len(self.index) / self.batch_size))

    def __getitem__(self, id):
        batch_ids = self.indices[id*self.batch_size: (id+1)*self.batch_size]
        if self.mode =='train':
            X, y = self._data_generation(batch_ids)
            return X, y
        else:
            X = self._data_generation(batch_ids)
            return X

    def on_epoch_end(self):
        self.indices = np.arange(len(self.available_ids))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _data_generation(self, batch_ids):
        height = self.input_shape[0]
        width = self.input_shape[1]
        chans = self.input_shape[2]
        X = np.empty((len(batch_ids), height, width, chans))
        y = np.empty((len(batch_ids), len(self.classes)))
        batch_diff = 0

        for i, id in enumerate(batch_ids):
            if self.mode=='train':
                fname = os.path.join(self.img_path, self.index.iloc[id]['label'] ,os.path.join(self.index.iloc[id]['fname']))
            else:
                fname = os.path.join(self.img_path, os.path.join(self.index.iloc[id]['fname']))
            try:
                img = Image.open(fname)

            except Exception as e:
                if not self.quiet:
                    print(f'loading file {fname} raised an exception {e}, this file was skipped')
                batch_diff += 1
                new_X = np.empty((len(batch_ids)-batch_diff, height, width, chans))
                new_y = np.empty((len(batch_ids)-batch_diff, len(self.classes)))
                new_X[:i-batch_diff+1] = X[:i-batch_diff+1]
                new_y[:i-batch_diff+1] = y[:i-batch_diff+1]
                X = new_X
                y = new_y
                continue

            if height != img.size[0] or self.width != img.size[1]:
                img = img.resize((height, width))

            if len(img.size)==2:
                extended_img = np.zeros((*img.size, chans))
                for chan in range(chans):
                    extended_img[:,:,chan] = img
                img = extended_img
            elif img.size[-1] < chans:
                extended_img = np.zeros((*img.size[:-1], chans))
                for chan in range(chans):
                    extended_img[:,:,chan] = img[:,:,0]
                img = extended_img

            img = np.asarray(img)

            if self.norm == 'sample':
                mean, var = tf.nn.moments(img, axes=[0,1])
                img = (img-mean.numpy())/np.sqrt(var.numpy()+1e-8)  # prevent 0 division
            X[i-batch_diff] = img
            if self.mode == 'train':
                label = self.index.iloc[id]['label']
                y[i-batch_diff] = (label == self.classes).astype('int')
        if self.norm == 'batch':
            mean, var = tf.nn.moments(X, axes=[0,1,2,3])
            X = (X-mean.numpy())/tf.sqrt(var.numpy()+1e-8)
        if self.mode =='train':
            return X,y
        else:
            return X
