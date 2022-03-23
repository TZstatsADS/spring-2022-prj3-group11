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

def create_model_II(train_clean_images, train_clean_labels, train_noisy_labels, noisy_images, noisy_labels, validation_split, epochs1, epochs2):
    print("updated2")
    # PHASE 1: Construct label cleaning network using images, and labels as inputs

    # Define input layers
    img_input = Input(shape=(32, 32, 3))
    noisy_label = Input(shape=(10))

    # Image feature extraction using pre-trained Resnet50
    resnet = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=(32, 32, 3),
        pooling='max'
    )

    # Image branch
    img_vec = resnet(img_input)
    img_vec = Dense(1024)(img_vec)
    img_vec = Dense(512)(img_vec)
    img_vec = Dense(256)(img_vec)

    # Noisy label branch
    noisy_l = Dense(10)(noisy_label)

    # Concatenate both the noisy label and the image vector
    x = Concatenate(axis=-1)([noisy_l, img_vec])
    x = Dense(256, activation='relu')(x)
    out = Dense(10, activation='softmax')(x)

    # Final model, combining everything
    model = Model([img_input, noisy_label], out)

    # Compile the model
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['acc'], optimizer=tf.keras.optimizers.Adam(0.001))

    # Fit the model using the clean images
    clean_labels_train = np.eye(10)[train_clean_labels]
    noisy_labels_train = np.eye(10)[train_noisy_labels]
    imgs_train = train_clean_images
    
    callbacks = [
        EarlyStopping(patience=4)
    ]

    model.fit([imgs_train, noisy_labels_train], clean_labels_train, batch_size=128, epochs=epochs1,
              validation_split=validation_split, callbacks=callbacks)

    # Predict the noisy labels, taking the maximum
    cleaned_labels = model.predict([noisy_images, np.eye(10)[noisy_labels]])
    cleaned_labels = [np.argmax(i) for i in cleaned_labels]

    # Reconstruct the dataset, using the cleaned data and predicted labels
    x_train = np.concatenate((train_clean_images, noisy_images))
    y_train = np.concatenate((train_clean_labels, np.array(cleaned_labels)))

    # PHASE 3: Train final model (same as Model I) on cleaned data
    y_train = tf.one_hot(y_train, depth=10)

    start_time = time.time()

    final_model = tf.keras.Sequential(
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

    final_model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                        loss=tf.keras.losses.CategoricalCrossentropy(),
                        metrics=['accuracy'])

    history = final_model.fit(x_train, y_train, batch_size=128, epochs=epochs2,
                              validation_split=validation_split, callbacks=callbacks)

    return final_model, history, model

def test_model_II(test_images, test_labels, modelII):

    predictions = modelII.predict(test_images)
    predictions = [np.argmax(i) for i in predictions]
    predictions = np.array(predictions)

    return accuracy_score(predictions, test_labels)

def save_model_II(modelII, foldername):

    modelII.save('../output/' + foldername)

def load_model_II(foldername):

    return keras.models.load_model("../output/" + foldername)
    
