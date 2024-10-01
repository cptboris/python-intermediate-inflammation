"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest


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
