"""Tests for statistics functions within the Model layer."""

import os
import numpy as np
import numpy.testing as npt
from unittest.mock import Mock
import pytest
import math

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



@pytest.mark.parametrize('data, expected_standard_deviation', [
    ([0, 0, 0], 0.0),
    ([1.0, 1.0, 1.0], 0),
    ([0.0, 2.0], 1.0)
])
def test_daily_standard_deviation(data, expected_standard_deviation):
    from inflammation.models import daily_std_dev
    result_data = daily_std_dev(data)
    npt.assert_approx_equal(result_data, expected_standard_deviation)

    
@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [1,1], [2,2], [3,3] ], [3,3]),
        ([ [1.0,1.0], [2.0,2.0], [3.0,3.0] ], [3.0,3.0]),
        ([ [10,1], [2,20], [3,3] ], [10,20]),
    ])
    
def test_daily_max_integers(test,expected):
    """Test that max function works for an array of positive integers."""
    from inflammation.models import daily_max
    # Assert the comparison of arrays
    npt.assert_array_equal(daily_max(test),expected)


@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [1,1], [2,2], [3,3] ], [1,1]),
        ([ [1.0,1.0], [2,2], [3,3] ], [1.0,1.0]),
        ([ [-1,1], [2,0], [3,3] ], [-1,0]),
    ])

def test_daily_min_integers(test,expected):
    """Test that min function works for an array of positive integers"""
    from inflammation.models import daily_min
    # Assert the comparison of arrays
    npt.assert_array_equal(daily_min(test),expected)
    
def test_daily_min_string():
    """Test for TypeError when passing strings"""
    from inflammation.models import daily_min

    with pytest.raises(TypeError):
        error_expected = daily_min([['Hello', 'there'], ['General', 'Kenobi']])


@pytest.mark.parametrize(
    "test, expected, expect_raises",
    [
        (
            'hello',
            None,
            TypeError,
        ),
        (
            3,
            None,
            TypeError,
        ),
        (
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]], 
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            None
        ),
        (
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]], 
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            None
        ),
        (
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]], 
            [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
            None
        ),
        (
            [[-1, -1, 1], [-2,float('nan'), 2], [float('nan'), float('nan'), 3]], 
            [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
            ValueError,
        ),
        (
            [[-1, -1, -1], [-2,-2, -2], [-3, -3, -3]], 
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ValueError
        ),
    ])

def test_patient_normalise(test, expected, expect_raises):
    """Test normalisation works for arrays of one and positive integers.
       Test with a relative and absolute tolerance of 0.01."""
    from inflammation.models import patient_normalise
    
    if expect_raises is not None:
        with pytest.raises(expect_raises):
            result = patient_normalise(np.array(test))
            npt.assert_allclose(result, np.array(expected), rtol=1e-2, atol=1e-2)
    else:
        result = patient_normalise(np.array(test))
        npt.assert_allclose(result, np.array(expected), rtol=1e-2, atol=1e-2)


def test_compute_data_mock_source():
    from inflammation.compute_data import analyse_data
    data_source = Mock()
    data_source.load_inflammation_data.return_value = [[[0,2,0]],
                                                       [[0,1,0]]]
    
    result = analyse_data(data_source)
    npt.assert_array_almost_equal(result, [0, math.sqrt(0.25) ,0])


def test_analyse_data():
    from pathlib import Path
    from inflammation.compute_data import analyse_data
    from inflammation.models import CSVDataSource
    
    path = Path.cwd() / "data"
    data_source = CSVDataSource(path)
    result = analyse_data(data_source)
    expected_output = [0.,0.22510286,0.18157299,0.1264423,0.9495481,0.27118211,
                       0.25104719,0.22330897,0.89680503,0.21573875,1.24235548,0.63042094,
                       1.57511696,2.18850242,0.3729574,0.69395538,2.52365162,0.3179312,
                       1.22850657,1.63149639,2.45861227,1.55556052,2.8214853,0.92117578,
                       0.76176979,2.18346188,0.55368435,1.78441632,0.26549221,1.43938417,
                       0.78959769,0.64913879,1.16078544,0.42417995,0.36019114,0.80801707,
                       0.50323031,0.47574665,0.45197398,0.22070227]
    npt.assert_almost_equal(result,expected_output)

@pytest.mark.parametrize('data,expected_output', [
    ([[[0, 1, 0], [0, 2, 0]]], [0, 0, 0]),
    ([[[0, 2, 0]], [[0, 1, 0]]], [0, math.sqrt(0.25), 0]),
    ([[[0, 1, 0], [0, 2, 0]], [[0, 1, 0], [0, 2, 0]]], [0, 0, 0])
    ],
    ids=['Two patients in same file', 'Two patients in different files', 'Two identical patients in two different files'])

def test_compute_standard_deviation_by_day(data, expected_output):
    from inflammation.models import compute_standard_deviation_by_day

    result = compute_standard_deviation_by_day(data)
    npt.assert_array_almost_equal(result, expected_output)

