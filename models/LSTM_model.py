"""
File includes the lstm model and wrapper functions
Author: Christopher Bülte
"""

import sys
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
import statsmodels.api as sm
from math import sqrt as math_sqrt
from sklearn.metrics import mean_squared_error
import pickle
import math
import random
from tqdm import tqdm

from utils.utils import *


def mask_features(features, miss_vals, no_features):
    """
    Method to mask features in the lstm model

    :param features: Matrix of true features
    :param miss_vals: Matrix of which values are already missing
    :param no_features: Number of features
    :return: Matrix of masked features, matrix of missing values, mask
    """
    # Reshape mask to flat array to choose random samples
    mask = np.zeros(shape=(features.shape[0], features.shape[1] * features.shape[2]))
    for sample in range(features.shape[0]):
        # Draw number of missing values from Binomial distribution
        p = np.random.uniform(0.2, 0.8)
        number_miss_vals = np.random.binomial(n=24 * no_features, p=p)
        idx_all = list(range(24 * no_features))  # all features masked
        # Shuffle indices
        np.random.shuffle(idx_all)
        idx = idx_all[:number_miss_vals]
        for mv in range(number_miss_vals):
            # Insert mask at the middle day of the week
            mask[sample, (2 * 24) * no_features + idx[mv]] = 1

    # Reshape mask into original shape
    mask = mask.reshape(features.shape)
    # Remove missing values from mask
    mask[np.array(miss_vals) == 1] = 0
    # Mask features
    features_masked = np.array(features)
    features_masked[mask == 1] = 0
    # Set miss vals to 1 in miss_vals
    miss_vals_masked = np.array(miss_vals)
    miss_vals_masked[mask == 1] = 1

    return features_masked, miss_vals_masked, mask



class Bi_LSTM(tf.keras.Model):
    """
    Class for bi-directional lstm
    """
    def __init__(self, no_features, countries, years, rate=0.1):
        """
        Initialize the model

        :param no_features: No features from time series
        :param rate: Dropout rate
        """
        super(Bi_LSTM, self).__init__()

        # Embedding layers
        self.embedding_country = tf.keras.layers.Embedding(countries.shape[0], 9)
        self.embedding_year = tf.keras.layers.Embedding(years.shape[0], 4)

        # LSTM Layer
        self.lstm_layer = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(no_features, return_sequences=True, dropout=rate))

        # Final Dense Layers
        self.final_layer = tf.keras.layers.Dense(no_features)

    def call(self, input_call):
        """
        Call model

        :param input_call:
        :return: Estimated output [batch_size, 24, no_features]
        """
        country, year, hour, weekday, features_masked, miss_vals_masked, pos_enc = input_call

        # Embeddings
        country_emb = self.embedding_country(country)
        year_emb = self.embedding_year(year)

        # concatenation (embeddings plus features)
        inp = tf.concat([pos_enc,
                         tf.repeat(country_emb, pos_enc.shape[1], axis=1),
                         year_emb,
                         features_masked,
                         miss_vals_masked], axis=2)

        # LSTM Layer
        lstm_output = self.lstm_layer(inp)
        # Final Dense Layer
        final_output = self.final_layer(lstm_output)

        return final_output


def train_step(lstm, optimizer, country, year, hour, weekday, features, features_masked, miss_vals_masked, pos_enc, mask):
    """
    Train step for lstm model

    :param country:
    :param year:
    :param hour:
    :param weekday:
    :param features:
    :param features_masked:
    :param miss_vals_masked:
    :param pos_enc:
    :param mask:
    :return: model loss and prediction
    """
    training = True
    with tf.GradientTape() as tape:
        pred = lstm([country, year, hour, weekday, features_masked, miss_vals_masked, pos_enc])
        loss = loss_function(features, pred, mask)

    gradients = tape.gradient(loss, lstm.trainable_variables)
    optimizer.apply_gradients(zip(gradients, lstm.trainable_variables))
    return loss, pred


def loss_function(real, pred, mask):
    """
    Loss function to evaluate model only on masked values

    :param real: Real values
    :param pred: Predicted values
    :param mask: Mask
    :return: loss
    """
    real = real[mask==1]
    pred = pred[mask==1]
    real = tf.dtypes.cast(real, tf.float32)
    error = real-pred
    error = tf.square(error)
    loss = tf.math.sqrt(tf.math.reduce_mean(error))*1000
    return loss