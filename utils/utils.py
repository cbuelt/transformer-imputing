import pickle
import numpy as np


def load_data():
    training_data = np.load("../data/training_data_reshaped.npy")
    testing_data = np.load("../data/testing_data_reshaped.npy")
    testing_index = np.load("../data/testing_data_index.npy", allow_pickle=True)

    return training_data, testing_data, testing_index


def load_data_information():
    with open("../data/data_information.pkl", "rb") as p:
        data_information = pickle.load(p)
    countries, years, feature_names, no_features = data_information
    hours = np.arange(24)
    months = np.arange(12)
    weekdays = np.arange(7)

    return countries, no_features, feature_names, years, months, weekdays, hours