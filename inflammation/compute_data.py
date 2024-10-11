"""Module containing mechanism for calculating standard deviation between datasets.
"""

import glob
import os
import numpy as np

from inflammation import models, views

class CSVDataSource:
    def __init__(self,data_dir):
        self.data_dir = data_dir
        
    def load_inflammation_data(self):
        """Loads all the inflammation data .csv files in a given directory
        returns them as a list of 2D numpy arrays."""
        data_file_paths = glob.glob(os.path.join(self.data_dir, 'inflammation*.csv'))
        if len(data_file_paths) == 0:
            raise ValueError(f"No inflammation data CSV files found in path {data_dir}")
        data = map(models.load_csv, data_file_paths)
        return list(data)


def analyse_data(data_source):
    """Calculates the standard deviation by day between datasets.

    Gets all the inflammation data from CSV files within a directory,
    works out the mean inflammation value for each day across all datasets,
    then plots the graphs of standard deviation of these means."""
    
    data = data_source.load_inflammation_data()

    daily_standard_deviation = compute_standard_deviation_by_day(data)
    graph_data = {
        'standard deviation by day': daily_standard_deviation,
    }
    # views.visualize(graph_data)
    return daily_standard_deviation


def compute_standard_deviation_by_day(data):
    """Calculates the daily standard deviation
    """
    means_by_day = map(models.daily_mean, data)
    means_by_day_matrix = np.stack(list(means_by_day))

    daily_standard_deviation = np.std(means_by_day_matrix, axis=0)
    return daily_standard_deviation
