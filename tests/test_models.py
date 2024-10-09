"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
from unittest.mock import Mock

def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    from inflammation.models import daily_mean

    test_input = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""
    from inflammation.models import daily_mean

    test_input = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])
    test_result = np.array([3, 4])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)

def test_compute_data_mock_source():
    from inflammation.compute_data import analyse_data
    import math
    data_source = Mock()
    data_source.load_inflammation_data.return_value = [[[0,2,0]],
                                                       [[0,1,0]]]
    
    result = analyse_data(data_source)
    npt.assert_array_almost_equal(result, [0, math.sqrt(0.25) ,0])
