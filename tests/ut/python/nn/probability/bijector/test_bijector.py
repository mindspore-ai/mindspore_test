# Copyright 2020 Huawei Technologies Co., Ltd
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
"""test cases for exp"""
import numpy as np
import pytest

import mindspore.nn as nn
import mindspore.nn.probability.bijector as msb
from mindspore import Tensor
from mindspore import dtype

class MyBijector(msb.Bijector):
    """
    Customized bijector class with dtype not specified.
    """
    def __init__(self, param1, param2):
        param = dict(locals())
        param['param_dict'] = {'param1': param1, 'param2': param2}
        super(MyBijector, self).__init__(name='MyBijector', dtype=None, param=param)

        self._param1 = self._add_parameter(param1, 'param1')
        self._param2 = self._add_parameter(param2, 'param2')

    @property
    def param1(self):
        return self._param1

    @property
    def param2(self):
        return self._param2

    def _forward(self, value):
        value = self._check_value_dtype(value)
        param1_local = self.cast_param_by_value(value, self.param1)
        param2_local = self.cast_param_by_value(value, self.param2)
        return value * param1_local + param2_local

class MySecondBijector(msb.Bijector):
    """
    Customized bijector class with dtype specified.
    """
    def __init__(self, param1, param2):
        param = dict(locals())
        param['param_dict'] = {'param1': param1, 'param2': param2}
        super(MySecondBijector, self).__init__(name='MySecondBijector', dtype=dtype.float32, param=param)

        self._param1 = self._add_parameter(param1, 'param1')
        self._param2 = self._add_parameter(param2, 'param2')

    @property
    def param1(self):
        return self._param1

    @property
    def param2(self):
        return self._param2

    def _forward(self, value):
        value = self._check_value_dtype(value)
        param1_local = self.cast_param_by_value(value, self.param1)
        param2_local = self.cast_param_by_value(value, self.param2)
        return value * param1_local + param2_local

class Net1(nn.Cell):
    """
    Test input value when bijector's dtype is not specified.
    """
    def __init__(self):
        super(Net1, self).__init__()
        self.bijector = MyBijector(np.array(1.0).astype(np.float32), np.array(2.0).astype(np.float32))

    def construct(self, value):
        return self.bijector.forward(value)

class Net2(nn.Cell):
    """
    Test input value when bijector's dtype is specified.
    """
    def __init__(self):
        super(Net2, self).__init__()
        self.bijector = MySecondBijector(np.array(1.0).astype(np.float32), np.array(2.0).astype(np.float32))

    def construct(self, value):
        return self.bijector.forward(value)

def test_input_value():
    """
    Test validity of input value.
    """
    net = Net1()
    value = None
    with pytest.raises(TypeError):
        ans = net(value)
    value = 1.0
    with pytest.raises(TypeError):
        ans = net(value)
    value = Tensor(1.0, dtype=dtype.int32)
    with pytest.raises(TypeError):
        ans = net(value)
    value = Tensor(1.0, dtype=dtype.float32)
    ans = net(value)
    assert ans.dtype == dtype.float32
    value = Tensor(1.0, dtype=dtype.float16)
    ans = net(value)
    assert ans.dtype == dtype.float16

def test_input_value2():
    """
    Test validity of input value.
    """
    net = Net2()
    value = None
    with pytest.raises(TypeError):
        ans = net(value)
    value = 1.0
    with pytest.raises(TypeError):
        ans = net(value)
    value = Tensor(1.0, dtype=dtype.int32)
    with pytest.raises(TypeError):
        ans = net(value)
    value = Tensor(1.0, dtype=dtype.float16)
    with pytest.raises(TypeError):
        ans = net(value)
    value = Tensor(1.0, dtype=dtype.float32)
    ans = net(value)
    assert ans.dtype == dtype.float32
 