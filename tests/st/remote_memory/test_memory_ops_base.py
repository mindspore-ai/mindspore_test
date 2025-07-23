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
import pytest
import numpy as np
from mindspore import jit, ops
from mindspore import Tensor
from tests.mark_utils import arg_mark

from mindspore import context
context.set_context(save_graphs=True, save_graphs_path="./ir")


@pytest.mark.skip(reason='wait for ops')
@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_remote_ops_to_remote():
    """
    Feature: Remote memory base operator
    Description: Base scene.
    Expectation: No Exception.
    """

    @jit
    def foo(x):
        x = ops.ToRemote()(x, )
        return x

    x = Tensor([1, 2, 3, 4])
    ret = foo(x)
    assert np.all(ret.asnumpy() == np.array((1, 2, 3, 4)))


@pytest.mark.skip(reason='wait for ops')
@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_remote_ops_detach():
    """
    Feature: Remote memory base operator
    Description: Base scene.
    Expectation: No Exception.
    """

    @jit
    def foo(x):
        x = ops.ToRemote()(x)
        x = ops.Detach()(x)
        return x

    x = Tensor([1, 2, 3, 4])
    ret = foo(x)
    assert np.all(ret.asnumpy() == np.array((1, 2, 3, 4)))


@pytest.mark.skip(reason='wait for ops')
@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_remote_ops_prefetch():
    """
    Feature: Remote memory base operator
    Description: Base scene.
    Expectation: No Exception.
    """

    @jit
    def foo(x):
        x = ops.ToRemote()(x)
        x = ops.Detach()(x)
        x = ops.Prefetch()(x)
        return x

    x = Tensor([1, 2, 3, 4])
    ret = foo(x)
    assert np.all(ret.asnumpy() == np.array((1, 2, 3, 4)))


@pytest.mark.skip(reason='wait for ops')
@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_remote_ops_grad_load_forward():
    """
    Feature: Remote memory base operator
    Description: Base scene.
    Expectation: No Exception.
    """

    @jit
    def foo(x):
        y = ops.relu(x)
        y = ops.GradLoad()(y, x)
        return y

    x = Tensor([1, 2, 3, 4])
    ret = foo(x)
    assert np.all(ret.asnumpy() == np.array((1, 2, 3, 4)))


@pytest.mark.skip(reason='wait for ops')
@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_remote_ops_grad_load_grad():
    """
    Feature: Remote memory base operator
    Description: Base scene.
    Expectation: No Exception.
    """

    def foo(x):
        y = ops.relu(x)
        y = ops.GradLoad()(y, x, (), False)
        return y
    
    @jit
    def grad_foo(x):
        return ops.grad(foo)(x)

    x = Tensor([1, 2, 3, 4])
    ret = grad_foo(x)
    assert np.all(ret.asnumpy() == np.array((1, 2, 3, 4)))
