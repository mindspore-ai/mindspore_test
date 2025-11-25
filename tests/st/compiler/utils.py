# coding=utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Utils for compiler tests"""

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


def assert_equal(expected, actual, decimal=7, err_msg=''):
    if isinstance(expected, (list, tuple)):
        assert type(expected) is type(actual)
        assert len(expected) == len(actual)
        for l, r in zip(expected, actual):
            assert_equal(l, r, decimal=decimal, err_msg=err_msg)
    elif isinstance(expected, dict):
        assert type(expected) is type(actual)
        assert len(expected) == len(actual)
        for k in expected:
            assert k in actual
            assert_equal(expected[k], actual[k], decimal=decimal, err_msg=err_msg)
    elif isinstance(expected, Tensor):
        assert isinstance(actual, Tensor)
        match_array(actual, expected, error=decimal, err_msg=err_msg)
    else:
        assert expected == actual, f'expect: {expected}, actual: {actual}'


def _count_unequal_element(data_expected, data_me, rtol, atol):
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_me) * rtol)
    nan_diff = np.not_equal(np.isnan(data_expected), np.isnan(data_me))
    inf_diff = np.not_equal(np.isinf(data_expected), np.isinf(data_me))
    # ICKTGQ
    if data_expected.dtype in ('complex64', 'complex128'):
        greater = greater + nan_diff + inf_diff
    else:
        neginf_diff = np.not_equal(np.isneginf(data_expected), np.isneginf(data_me))
        greater = greater + nan_diff + inf_diff + neginf_diff
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, \
        "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}". \
            format(data_expected[greater], data_me[greater], error[greater])


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    if not np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan):
        _count_unequal_element(data_expected, data_me, rtol, atol)
    else:
        assert np.array(data_expected).shape == np.array(data_me).shape
