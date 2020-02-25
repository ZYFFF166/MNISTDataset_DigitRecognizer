#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import numpy as np

import tensorflow as tf

from tensorflow.keras import datasets, layers, models

# 42000
train_data_path = 'train.csv'
with open(train_data_path) as f:
    reader = csv.reader(f)
    train_head_row = next(reader)
    train_labels_data_list = list()
    train_images_data_list = list()
    for row in reader:
        train_labels_data_list.append(row[0])
        train_images_data_list.append(row[1:])
    train_labels_data = np.array(train_labels_data_list).astype(np.uint8)
    train_images_data = np.array(train_images_data_list).astype(np.float32)
    train_images_data = train_images_data.reshape((42000, 28, 28, 1))


train_labels = train_labels_data
train_images = train_images_data

# set features in range of [0, 1]
train_images = train_images / 255.0

# 3. add convolutional layers
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.summary() # show the configuration of the model

# 4. add dense layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary() # show the configuration of the model

# 5. compile and train the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)


# 6. save the weights of the model
model.save_weights('./save_weights/my_save_weights')

