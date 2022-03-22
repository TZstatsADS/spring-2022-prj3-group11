# Import required packages
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

import time
import tensorflow as tf
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Flatten, MaxPooling2D, Activation, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, Subtract
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Nadam
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import zipfile
from tensorflow.keras import layers
from tensorflow.keras import Model
import keras

def create_model_I(train_images, train_labels, validation_split, epochs):
    print("updated")
    x_train = train_images
    y_train = tf.one_hot(train_labels, depth=10)

    modelI = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255),

            tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation="relu", input_shape=(32, 32, 3)),
            tf.keras.layers.MaxPooling2D((2, 2), strides=2),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2), strides=2),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax")

        ]
    )

    modelI.compile(optimizer=tf.keras.optimizers.Nadam(0.001),
                   loss=tf.keras.losses.CategoricalCrossentropy(),
                   metrics=['accuracy'])

    callbacks = [
        EarlyStopping(patience=2)
    ]

    history = modelI.fit(x_train, y_train, batch_size=128, epochs=epochs,
                         validation_split=validation_split, callbacks=callbacks)

    return modelI, history

def test_model_I(test_images, test_labels, modelI):

    predictions = modelI.predict(test_images)
    predictions = [np.argmax(i) for i in predictions]
    predictions = np.array(predictions)

    return accuracy_score(predictions, test_labels)

def save_model_I(modelI, foldername):

    modelI.save('../output/' + foldername)

def load_model_I(foldername):

    return keras.models.load_model("../output/" + foldername)