# coding=utf-8

import numpy as np
from mindspore import Tensor


def match_array(actual, expected, error=0, err_msg=''):
    if isinstance(actual, (int, tuple, list, bool)):
        actual = np.asarray(actual)
    if isinstance(actual, Tensor):
        actual = actual.asnumpy()
    if isinstance(expected, (int, tuple, list, bool)):
        expected = np.asarray(expected)
    if isinstance(expected, Tensor):
        expected = expected.asnumpy()
    if error > 0:
        np.testing.assert_almost_equal(
            actual, expected, decimal=error, err_msg=err_msg)
    else:
        np.testing.assert_equal(actual, expected, err_msg=err_msg)
