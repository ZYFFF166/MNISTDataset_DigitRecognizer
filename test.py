#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# 1. build the model
# 1.1. add convolutional layers
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.summary()  # show the configuration of the model

# 1.2. add dense layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()  # show the configuration of the model

# 2. get weights
model.load_weights('./save_weights/my_save_weights')

# 28000
test_data_path = 'test.csv'
with open(test_data_path) as f:
    reader = csv.reader(f)
    test_head_row = next(reader)
    test_images_data = [row for row in reader]
    test_images_data = np.array(test_images_data).astype(np.float32)
    test_images_data = test_images_data.reshape((28000, 28, 28, 1))
test_images = test_images_data

# set features in range of [0, 1]
test_images = test_images / 255.0

# test
# test_loss, test_acc = model.evaluate(test_images, test_labels)

# predict
test_labels = np.argmax(model.predict(test_images), axis=1).astype(np.uint8)
# print(test_labels)
list = []
for i in range(1,len(test_labels)+1):
    list.append(i)
dataframe = pd.DataFrame({'ImageId':list,'Label':test_labels})
# np.savetxt('./test_labels.csv', test_labels, delimiter=',', fmt='%d')
dataframe.to_csv("test_submission.csv",index=False,sep=',')
