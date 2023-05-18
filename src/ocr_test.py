from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests
import os
import skimage
import numpy as np 
import tensorflow as tf
from dataloader import DataLoader 
import tqdm
# load image from the IAM database
# url = 'https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg'
# image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# data loading

def decode_img(img):
    # reading image from filepath
    I = skimage.io.imread(img)

    # check if it is in vertical orientatioon
    if I.shape[0] < I.shape[1]:
        I = I.T
    # add channel dim
    I = np.expand_dims(I, axis=-1)
    # resize and return
    return tf.image.resize(I, [224, 224])


if __name__ == '__main__':
    # loading data
    loader = DataLoader()
    images = loader.load('IMG_TEST')

    # data = []
    # path = "../data/datasets/train_set/"
    # classes = os.listdir(path)

    # for class_name in tqdm(classes):
    #     files = os.listdir(path + class_name)
    #     class_images = np.zeros((len(files), 224, 224,1))
    #     for i, file in enumerate(files):
    #         I = decode_img(path + class_name + "/" + file)
    #         class_images[i] = I
    #     data.append(class_images)

    # print(len(data))
    # for x in data:
    #     print(x.shape)

    # print("Done")

# processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
# model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
# pixel_values = processor(images=image, return_tensors="pt").pixel_values

# generated_ids = model.generate(pixel_values)
# generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# print(generated_text)