"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.

Patients' data is held in an inflammation table (2D array) where each row contains 
inflammation data for a single patient taken over a number of days 
and each column represents a single day across all patients.
"""

import numpy as np
import glob
import os

class JSONDataSource:
    def __init__(self,filename):
        self.filename = filename
        
    def load_inflammation_data(self):
        data_file_paths = glob.glob(os.path.join(self.dir_path, 'inflammation*.json'))
        if len(data_file_paths) == 0:
            raise ValueError(f"No inflammation JSON files found in path {self.dir_path}")
        data = map(models.load_json, data_file_paths)
        return list(data)

class CSVDataSource:
    def __init__(self,data_dir):
        self.data_dir = data_dir
        
    def load_inflammation_data(self):
        """Loads all the inflammation data .csv files in a given directory
        returns them as a list of 2D numpy arrays."""
        data_file_paths = glob.glob(os.path.join(self.data_dir, 'inflammation*.csv'))
        if len(data_file_paths) == 0:
            raise ValueError(f"No inflammation data CSV files found in path {data_dir}")
        data = map(load_csv, data_file_paths)
        return list(data)

def load_csv(filename):
    """Load a Numpy array from a CSV

    :param filename: Filename of CSV to load
    """
    return np.loadtxt(fname=filename, delimiter=',')


def daily_mean(data):
    """Calculate the daily mean of a 2D inflammation data array.
    param data: 2D array containing inflammation data
    returns: an array of mean values of data along the first (0-zero) axis
    """
    return np.mean(data, axis=0)


def daily_max(data):
    """Calculate the daily max of a 2D inflammation data array.
    param data: 2D array containing inflammation data
    returns: an array of maxima of data along the first (0-zero) axis
    """
    return np.max(data, axis=0)


def daily_min(data):
    """Calculate the daily min of a 2D inflammation data array.
    param data: 2D array containing inflammation data
    returns: an array of minima of data along the first (0-zero) axis
    """
    return np.min(data, axis=0)



def daily_std_dev(data):
    """Computes and returns standard deviation for data."""
    return np.std(data,axis=0)

  
def patient_normalise(data):
    """
    Normalise patient data from a 2D inflammation data array.

    NaN values are ignored, and normalised to 0.

    Negative values are rounded to 0.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError('data input should be ndarray')
    if len(data.shape) != 2:
        raise TypeError('inflammation array should be 2-dimensional')
    if np.any(data < 0):
        raise ValueError('Inflammation values should not be negative')
    max_data = np.nanmax(data, axis=1)
    max_data = np.where( max_data <= 0.0, 1.0, max_data )
    with np.errstate(invalid='ignore', divide='ignore'):
        normalised = data / max_data[:, np.newaxis]
    normalised[np.isnan(normalised)] = 0
    normalised[normalised < 0] = 0
    return normalised


def compute_standard_deviation_by_day(data):
    """Calculates the daily standard deviation
    """
    means_by_day = map(daily_mean, data)
    means_by_day_matrix = np.stack(list(means_by_day))

    daily_standard_deviation = np.std(means_by_day_matrix, axis=0)
    return daily_standard_deviation

