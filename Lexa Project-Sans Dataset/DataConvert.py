import numpy as np
import os
import PIL
import tensorflow as tf
import pathlib
directory = "C:/Users/rudy_/PycharmProjects/pythonProject/Classified"

(images, labels) = tf.keras.preprocessing.image_dataset_from_directory(
    directory,
    labels="inferred",
    label_mode='binary',
    class_names=['Genuine', 'Replayed'],
    color_mode="rgb",
    batch_size=32,
    image_size=(1182, 597),
    shuffle=True,
    seed=158,
    validation_split=.2,
    subset='training',
    interpolation="bilinear",
    follow_links=False
    )
validation = tf.keras.preprocessing.image_dataset_from_directory(
    directory,
    labels="inferred",
    label_mode="binary",
    color_mode="rgb",
    batch_size=32,
    image_size=(1182, 597),
    shuffle=True,
    seed=158,
    validation_split=.2,
    subset='validation',
    interpolation="bilinear",
    follow_links=False
)

(images, labels) = training

print(images)