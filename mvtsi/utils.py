"""
Utility functions
Author: Christopher Bülte
"""

import pickle
import numpy as np
import tensorflow as tf
import random



def load_data():
    """
    Load prepared training-, testing-data and the time index of the testing-data
    :return: Tuple of numpy arrays
    """
    training_data = np.load("../data/training_data_reshaped.npy")
    testing_data = np.load("../data/testing_data_reshaped.npy")
    testing_index = np.load("../data/testing_data_index.npy", allow_pickle=True)
    return training_data, testing_data, testing_index


def load_data_information():
    """
    Load data information from pickle
    :return: Tuple of information about the data
    """
    with open("../data/data_information.pkl", "rb") as p:
        data_information = pickle.load(p)
    countries, years, feature_names, nu_features = data_information
    hours = np.arange(24)
    months = np.arange(12)
    weekdays = np.arange(7)
    return countries, nu_features, feature_names, years, months, weekdays, hours


# Extracts features, year, etc. from whole data
def change_format(input_data, nu_features):
    '''
    Transforms the raw input data into single vectors containing the different information
    (e.g. country, values, feature_nr...)
    :param input_data: Input data in raw format
    :param nu_features: Number of available features
    :return: features_tf, miss_vals_tf, pos_enc_tf, country_w, year_tf, feature_nr_tf, prob_mask
    '''
    # Extract year from data matrix
    year_w = input_data[:, :, 0:1]
    # Extract weekday from data matrix
    weekday_w = input_data[:, :, 1:2]
    # Extract hour from data matrix
    hour_w = input_data[:, :, 2:3]
    # Extract country from data matrix
    country_w = input_data[:, 0:1, 3]
    # Extract month from data matrix
    month_w = input_data[:, :, 4:5]
    # Extract features from matrix
    features_w = input_data[:, :, 5:5 + nu_features]
    # Extract matrix of missing values from data matrix
    miss_vals_w = input_data[:, :, 5 + nu_features:5 + 2 * nu_features]
    # Extract pos enc from data matrix
    pos_enc_w = input_data[:, :, 5 + 2 * nu_features:11 + 2 * nu_features]
    # Extract Probability mask
    prob_mask = input_data[:, :, -nu_features:]

    # Prepare format for features
    features_tf = np.reshape(features_w, [features_w.shape[0], -1, 1])
    miss_vals_tf = np.reshape(miss_vals_w, [features_w.shape[0], -1, 1])
    pos_enc_tf = np.reshape(tf.transpose(
        np.repeat(np.reshape(pos_enc_w, [pos_enc_w.shape[0], pos_enc_w.shape[1], pos_enc_w.shape[2], 1]), nu_features,
                  axis=3), perm=[0, 1, 3, 2]), [pos_enc_w.shape[0], -1, pos_enc_w.shape[2]])
    feature_nr_tf = np.repeat(
        np.reshape(np.repeat(np.reshape(np.array(range(nu_features)), [1, -1]), input_data.shape[1], axis=0), [1, -1]),
        input_data.shape[0], axis=0)

    # Reshape other features
    hour_tf = np.reshape(np.repeat(hour_w, nu_features, axis=2), [input_data.shape[0], -1])
    year_tf = np.reshape(np.repeat(year_w, nu_features, axis=2), [input_data.shape[0], -1])
    weekday_tf = np.reshape(np.repeat(weekday_w, nu_features, axis=2), [input_data.shape[0], -1])
    month_tf = np.reshape(np.repeat(month_w, nu_features, axis=2), [input_data.shape[0], -1])

    return features_tf, miss_vals_tf, pos_enc_tf, country_w, year_tf, feature_nr_tf, prob_mask


def mask_features(features, miss_vals, prob_mask):
    '''
    Mask the features, based on the probability mask, created by the first stage mdoel
    :param features: Feature vector
    :param miss_vals: Missing value vector
    :param prob_mask: Probability mask (probability that one entry is missing)
    :return: features_masked, miss_vals_masked, realized_mask, miss_vals
    '''
    #Create and reshape mask
    realized_mask = np.zeros(shape = prob_mask.shape)
    realized_mask[:,2*24:3*24] = np.random.binomial(1, prob_mask[:,2*24:3*24])
    realized_mask = realized_mask.reshape(features.shape)

    #Mask features
    features_masked = np.array(features)
    features_masked[realized_mask == 1] = 0
    #Mask missing values
    miss_vals_masked = np.array(miss_vals)
    miss_vals_masked[realized_mask == 1] = 1

    return features_masked, miss_vals_masked, realized_mask, miss_vals


def val_split(data, split=0.2):
    '''
    Create validation split
    :param data: Data in required format
    :param split: Split ratio
    :return: data_train, data_val
    '''
    size = int(data.shape[0] * split)
    index = data.shape[0]
    split_index = random.choices(range(index), k=size)
    data_val = data[split_index]
    data_train = np.delete(data, split_index, axis=0)
    return data_train, data_val
