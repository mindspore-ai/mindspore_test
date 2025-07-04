# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Test Tensor check_input_data_type"""
import pytest
import numpy as np
import mindspore as ms
from mindspore import Tensor


def test_convert_to_tensor_by_structured_array():
    """
    Feature: Check the type of input_data for Tensor.
    Description: Convert to Tensor by structured array.
    Expectation: Throw TypeError.
    """
    a = np.array([('x', 1), ('y', 2)], dtype=[('name', '<U10'), ('value', '<i4')])
    with pytest.raises(TypeError) as ex:
        Tensor(a)
    assert "initializing tensor by numpy array failed" in str(ex.value)
    assert "<class 'numpy.void'>" in str(ex.value)


def test_convert_to_tensor_by_object_type_array():
    """
    Feature: Check the type of input_data for Tensor.
    Description: Convert to Tensor by object type.
    Expectation: Throw TypeError.
    """
    a = np.array([[1, 2, 3], [4, Tensor(5), 6], [7, 8, 9]], dtype=object)
    with pytest.raises(TypeError) as ex:
        Tensor(a)
    assert "initializing tensor by numpy array failed" in str(ex.value)


def test_convert_to_tensor_by_sequence():
    """
    Feature: Check the shape of input_data for Tensor.
    Description: Convert to Tensor by sequence type.
    Expectation: Throw ValueError.
    """

    with pytest.raises(TypeError) as ex:
        Tensor([[3], [[4]]])
    assert "For Tensor, the input_data is [[3], [[4]]] that contain unsupported element." in str(ex.value)

    with pytest.raises(TypeError) as ex:
        Tensor([[[3]], [4]])
    assert "For Tensor, the input_data is [[[3]], [4]] that contain unsupported element." in str(ex.value)

    with pytest.raises(TypeError) as ex:
        Tensor([[3], [4, 5]])
    assert "For Tensor, the input_data is [[3], [4, 5]] that contain unsupported element." in str(ex.value)

    with pytest.raises(TypeError) as ex:
        Tensor([[3], [[4, 5]]])
    assert "For Tensor, the input_data is [[3], [[4, 5]]] that contain unsupported element." in str(ex.value)

    with pytest.raises(TypeError) as ex:
        Tensor([[[3]], [[4, 5]]])
    assert "For Tensor, the input_data is [[[3]], [[4, 5]]] that contain unsupported element." in str(ex.value)

    with pytest.raises(TypeError) as ex:
        Tensor([[], [4]])
    assert "For Tensor, the input_data is [[], [4]] that contain unsupported element." in str(ex.value)

    with pytest.raises(TypeError) as ex:
        Tensor([[4], []])
    assert "For Tensor, the input_data is [[4], []] that contain unsupported element." in str(ex.value)

    with pytest.raises(TypeError) as ex:
        Tensor([[], None])
    assert "For Tensor, the input_data is [[], None] that contain unsupported element." in str(ex.value)

    with pytest.raises(TypeError) as ex:
        Tensor([None, []])
    assert "For Tensor, the input_data is [None, []] that contain unsupported element." in str(ex.value)


def test_tensor_with_empty_shape():
    """
    Feature: Check the empty shape of input_data for Tensor.
    Description: Create tensor with empty shape.
    Expectation: Success.
    """
    a = Tensor([])
    assert a.shape == (0,)
    assert a.dtype == ms.float32
    a = Tensor([[[], []], [[], []]], dtype=ms.int32)
    assert a.shape == (2, 2, 0)
    assert a.dtype == ms.int32
    a = Tensor(np.ones(shape=(2, 0, 3), dtype=np.float16))
    assert a.shape == (2, 0, 3)
    assert a.dtype == ms.float16
    a = Tensor(np.array([[[], []], [[], []], [[], []]]), dtype=ms.uint8)
    assert a.shape == (3, 2, 0)
    assert a.dtype == ms.uint8
    a = Tensor(shape=(1, 0, 3), dtype=ms.bfloat16, init=ms.common.initializer.One())
    assert a.shape == (1, 0, 3)
    assert a.dtype == ms.bfloat16
