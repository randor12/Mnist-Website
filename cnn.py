import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

class CNN():
    def __init__(self):
        """
        Initialize the CNN model
        """
        # initialize the values
        self.model = models.Sequential()
        # add the hidden layers to the CNN
        self.model.add(layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=(28, 28, 1)))  # create the convolutions
        self.model.add(layers.MaxPooling2D((2, 2)))  # use max pooling
        self.model.add(layers.Conv2D(64, 3, padding='same', activation='relu'))  # more layers ...
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(128, 3, padding='same', activation='relu'))
        self.model.add(layers.Dropout(0.1))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dense(10, activation='softmax'))  # get the softmax

        self.model.load_weights('cnn.h5')

    def predict(self, data):
        """
        Make a classification prediction from the data
        :param data: data to predict on
        :returns: returns the classification prediction(s) [0-9]
        """
        probs = self.model.predict(data)
        predictions = np.argmax(probs, axis=1)
        return predictions
