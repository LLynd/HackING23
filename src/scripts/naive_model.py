import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import preprocessing

PATH = "F:\Studia\INGHackathon\HackING23\src\data\datasets\\wffts"


PATTERN = "\*.npy"

CLASS_NUMBER = 21
SIZE = (400, 300)
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(21))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])



directory_list = os.listdir(PATH)
labels = np.array([])
imgs_array = np.zeros((0, *SIZE),dtype=np.uint16)
for i, class_dir in enumerate(directory_list):
    _path = os.path.join(PATH, class_dir)

    img_collection = os.listdir(_path)

    class_array = np.ones(len(img_collection))*i
    labels = np.concatenate((labels, class_array))
    print(class_dir)
    for fname in tqdm(img_collection):
        if not fname.endswith(".npy"):
            continue
        img_path = os.path.join(_path, fname)
        img = np.load(img_path)
        imgs_array = np.concatenate((imgs_array, img))


X_train, X_test, y_train, y_test = train_test_split(imgs_array, labels, test_size=0.33, random_state=42)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

le = preprocessing.LabelEncoder()
le.fit(y_train)
y_train = le.transform(y_train)
y_test = le.transform(y_test)

history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_test, y_test))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.savefig("acc.png")