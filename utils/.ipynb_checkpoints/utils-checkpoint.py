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


def get_dictionaries():
    country_dict = {'AT': 0,
                     'BE': 1,
                     'BG': 2,
                     'CH': 3,
                     'CZ': 4,
                     'DE-AT-LU': 5,
                     'DE-LU': 6,
                     'DK': 7,
                     'EE': 8,
                     'ES': 9,
                     'FI': 10,
                     'FR': 11,
                     'GB': 12,
                     'GR': 13,
                     'HR': 14,
                     'HU': 15,
                     'IE': 16,
                     'IT': 17,
                     'LT': 18,
                     'LV': 19,
                     'NL': 20,
                     'NO': 21,
                     'PL': 22,
                     'PT': 23,
                     'RO': 24,
                     'RS': 25,
                     'SE': 26,
                     'SI': 27,
                     'SK': 28}
    
    feature_dict = {'Day_ahead_price': 0,                    
                     'Load': 1,
                     'Biomass': 2,
                     'Fossil Gas': 3,
                     'Hard coal': 4,
                     'Pumped Storage': 5,
                     'Other': 6,
                     'PV': 7,
                     'Wind Onshore': 8,
                     'Lignite': 9,
                     'Nuclear': 10,
                     'Wind Offshore': 11,
                     'Hydro': 12}
    
    return (country_dict, feature_dict)