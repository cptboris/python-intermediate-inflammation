"""Module containing mechanism for calculating standard deviation between datasets.
"""

import numpy as np

from inflammation import models, views


def analyse_data(data_source):
    """Calculates the standard deviation by day between datasets.

    Gets all the inflammation data from CSV files within a directory,
    works out the mean inflammation value for each day across all datasets,
    then plots the graphs of standard deviation of these means."""
    
    data = data_source.load_inflammation_data()

    daily_standard_deviation = models.compute_standard_deviation_by_day(data)
    return daily_standard_deviation


