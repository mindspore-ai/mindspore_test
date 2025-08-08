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
import os
import pytest
import numpy as np
import mindspore as ms
from mindspore import mutable
from mindspore import jit, ops
from mindspore import Tensor, Parameter
from mindspore.nn import Cell
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_remote_ops_copy_to_remote():
    """
    Feature: Remote memory base operator
    Description: Base scene.
    Expectation: No Exception.
    """

    @jit
    def foo(x):
        x = ops.auto_generate.CopyToRemote()(x)
        return x

    x = Tensor([1, 2, 3, 4])
    ret = foo(x)
    assert np.all(ret.asnumpy() == np.array((1, 2, 3, 4)))
    assert ret.device == "CPU"

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_remote_ops_copy_to():
    """
    Feature: Remote memory base operator
    Description: Base scene.
    Expectation: No Exception.
    """
    ms.set_context(device_id=0)

    @jit
    def foo(x):
        x = ops.auto_generate.CopyToRemote()(x)
        x = ops.auto_generate.CopyToDevice()(x)
        return x

    x = Tensor([1, 2, 3, 4])
    ret = foo(x)
    assert np.all(ret.asnumpy() == np.array((1, 2, 3, 4)))
    assert ret.device == "Ascend:0"

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_remote_ops_copy_to_and_free():
    """
    Feature: Remote memory base operator
    Description: Base scene.
    Expectation: Throw RuntimeError.
    """
    ms.set_context(device_id=0)

    @jit
    def foo(x):
        y = ops.auto_generate.CopyToRemote()(x)
        a = ops.auto_generate.FreeDevice()(x)
        b = ops.depend(y, a)
        z = ops.auto_generate.CopyToRemote()(x, b)
        return b, z

    with pytest.raises(RuntimeError) as err:
        x = Tensor([1, 2, 3, 4])
        y, z = foo(x)
        print(f'y: {y}, device of z: {z.device}')
    assert "The first input of Primitive 'CopyToRemote' has been released before." in str(err.value)

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_remote_ops_to_remote():
    """
    Feature: Remote memory base operator
    Description: Base scene.
    Expectation: No Exception.
    """
    os.environ['MS_DEV_ENABLE_REMOTE_MEMORY'] = '1'

    @jit
    def foo(x):
        x = ops.auto_generate.ToRemote()(x,)
        return x

    x = Tensor([1, 2, 3, 4])
    ret = foo(x)
    assert np.all(ret.asnumpy() == np.array((1, 2, 3, 4)))
    os.environ['MS_DEV_ENABLE_REMOTE_MEMORY'] = '0'


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_remote_ops_detach():
    """
    Feature: Remote memory base operator
    Description: Base scene.
    Expectation: No Exception.
    """
    os.environ['MS_DEV_ENABLE_REMOTE_MEMORY'] = '1'

    @jit
    def foo(x):
        x = ops.auto_generate.ToRemote()(x)
        x = ops.auto_generate.Detach()(x)
        return x

    x = Tensor([1, 2, 3, 4])
    ret = foo(x)
    assert np.all(ret.asnumpy() == np.array((1, 2, 3, 4)))
    os.environ['MS_DEV_ENABLE_REMOTE_MEMORY'] = '0'


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_remote_ops_prefetch():
    """
    Feature: Remote memory base operator
    Description: Base scene.
    Expectation: No Exception.
    """
    os.environ['MS_DEV_ENABLE_REMOTE_MEMORY'] = '1'

    @jit
    def foo(x):
        x = ops.auto_generate.ToRemote()(x)
        x = ops.auto_generate.Detach()(x)
        x = ops.auto_generate.Prefetch()(x)
        return x

    x = Tensor([1, 2, 3, 4])
    ret = foo(x)
    assert np.all(ret.asnumpy() == np.array((1, 2, 3, 4)))
    os.environ['MS_DEV_ENABLE_REMOTE_MEMORY'] = '0'


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_remote_functional_remote_memory():
    """
    Feature: Remote memory base operator
    Description: Base scene.
    Expectation: No Exception.
    """
    os.environ['MS_DEV_ENABLE_REMOTE_MEMORY'] = '1'

    @jit
    def foo(x):
        x = ops.auto_generate.to_remote(x)
        x = ops.auto_generate.detach(x)
        x = ops.auto_generate.prefetch(x)
        return x

    x = Tensor([1, 2, 3, 4])
    ret = foo(x)
    assert np.all(ret.asnumpy() == np.array((1, 2, 3, 4)))
    os.environ['MS_DEV_ENABLE_REMOTE_MEMORY'] = '0'


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_remote_ops_grad_load_forward():
    """
    Feature: Remote memory base operator
    Description: Base scene.
    Expectation: No Exception.
    """
    os.environ['MS_DEV_ENABLE_REMOTE_MEMORY'] = '1'

    @jit
    def foo(x):
        y = ops.relu(x)
        y = ops.auto_generate.GradLoad()(y, x)
        return y

    x = Tensor([1, 2, 3, 4], dtype=ms.int32)
    ret = foo(x)
    assert np.all(ret.asnumpy() == np.array((1, 2, 3, 4)))
    os.environ['MS_DEV_ENABLE_REMOTE_MEMORY'] = '0'


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_remote_ops_grad_load_grad():
    """
    Feature: Remote memory base operator
    Description: Base scene.
    Expectation: No Exception.
    """
    os.environ['MS_DEV_ENABLE_REMOTE_MEMORY'] = '1'

    def foo(x):
        y = ops.relu(x)
        y = ops.auto_generate.GradLoad()(y, x, (), False)
        return y

    @jit
    def grad_foo(x):
        return ops.grad(foo)(x)

    x = Tensor([1, 2, 3, 4], dtype=ms.int32)
    ret = grad_foo(x)
    assert np.all(ret.asnumpy() == np.array((1, 1, 1, 1)))
    os.environ['MS_DEV_ENABLE_REMOTE_MEMORY'] = '0'


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_remote_functional_grad_load_grad():
    """
    Feature: Remote memory base operator
    Description: Base scene.
    Expectation: No Exception.
    """
    os.environ['MS_DEV_ENABLE_REMOTE_MEMORY'] = '1'

    def foo(x):
        y = ops.relu(x)
        y = ops.auto_generate.grad_load(y, x, (), False)
        return y

    @jit
    def grad_foo(x):
        return ops.grad(foo)(x)

    x = Tensor([1, 2, 3, 4], dtype=ms.int32)
    ret = grad_foo(x)
    assert np.all(ret.asnumpy() == np.array((1, 1, 1, 1)))
    os.environ['MS_DEV_ENABLE_REMOTE_MEMORY'] = '0'


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_remote_ops_in_for_loop():
    """
    Feature: Remote memory base operator
    Description: Base scene.
    Expectation: No Exception.
    """
    os.environ['MS_DEV_ENABLE_REMOTE_MEMORY'] = '1'

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.param_a = Parameter(Tensor([1, 1, 1]), name="param_a")
            self.param_b = Parameter(Tensor([1, 1, 1]), name="param_b")
            self.param_c = Parameter(Tensor([1, 1, 1]), name="param_c")
            self.params = self.trainable_params()
            self.prefetch = ops.auto_generate.Prefetch()
            self.depend = ops.Depend()
            self.detach = ops.auto_generate.Detach()

        @jit
        def construct(self, a, b, c):
            m = (a, b, c)
            a = 0
            for i in range(3):
                prefetch_result = self.prefetch(self.params[i], sync=False)
                cur = self.depend(m[i], prefetch_result)
                a = a + cur + self.params[i]
                detach_result = self.detach(m[i], sync=False)
                a = self.depend(a, detach_result)
            return a

    x = Tensor([1, 1, 1])
    y = Tensor([1, 1, 1])
    z = Tensor([1, 1, 1])
    net = Net()
    ret = net(x, y, z)
    assert np.all(ret.asnumpy() == np.array((6, 6, 6)))
    os.environ['MS_DEV_ENABLE_REMOTE_MEMORY'] = '0'


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_remote_ops_in_for_loop_grad():
    """
    Feature: Remote memory base operator
    Description: Base scene.
    Expectation: No Exception.
    """
    os.environ['MS_DEV_ENABLE_REMOTE_MEMORY'] = '1'

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.param_a = Parameter(Tensor([1, 1, 1], dtype=ms.int32), name="param_a")
            self.param_b = Parameter(Tensor([1, 1, 1], dtype=ms.int32), name="param_b")
            self.param_c = Parameter(Tensor([1, 1, 1], dtype=ms.int32), name="param_c")
            self.params = self.trainable_params()
            self.prefetch_params = (self.param_b, self.param_c, None)
            self.prefetch = ops.auto_generate.Prefetch()
            self.depend = ops.Depend()
            self.detach = ops.auto_generate.Detach()

        @jit
        def construct(self, a, b, c):
            m = (a, b, c)
            a = 0
            for i in range(3):
                if self.prefetch_params[i] is None:
                    prefetch_result = None
                else:
                    prefetch_result = self.prefetch(self.prefetch_params[i], sync=False)
                cur = self.depend(m[i], prefetch_result)
                a = ops.relu(a + cur + self.params[i])
                detach_result = self.detach(m[i], sync=False)
                a = self.depend(a, detach_result)
            return a

    class GradNet(Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.weights = net.trainable_params()
            self.net = net

        @jit
        def construct(self, *inputs):
            return ops.grad(self.net, weights=self.weights)(*inputs)


    x = Tensor([1, 1, 1], dtype=ms.int32)
    y = Tensor([1, 1, 1], dtype=ms.int32)
    z = Tensor([1, 1, 1], dtype=ms.int32)
    net = GradNet(Net())
    net(x, y, z)
    os.environ['MS_DEV_ENABLE_REMOTE_MEMORY'] = '0'


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_remote_ops_in_for_variable_loop():
    """
    Feature: Remote memory base operator
    Description: Base scene.
    Expectation: No Exception.
    """
    os.environ['MS_DEV_ENABLE_REMOTE_MEMORY'] = '1'

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.param_a = Parameter(Tensor([1, 1, 1], dtype=ms.int32), name="param_a")
            self.param_b = Parameter(Tensor([1, 1, 1], dtype=ms.int32), name="param_b")
            self.param_c = Parameter(Tensor([1, 1, 1], dtype=ms.int32), name="param_c")
            self.params = self.trainable_params()
            self.prefetch = ops.auto_generate.Prefetch()
            self.depend = ops.Depend()
            self.detach = ops.auto_generate.Detach()

        @jit
        def construct(self, a, b, c, d):
            m = (a, b, c)
            a = Tensor([0, 0, 0])
            for i in range(d):
                prefetch_result = self.prefetch(self.params[i], sync=False)
                cur = self.depend(m[i], prefetch_result)
                a = a + cur + self.params[i]
                detach_result = self.detach(m[i], sync=False)
                a = self.depend(a, detach_result)
            return a

    x = Tensor([1, 1, 1], dtype=ms.int32)
    y = Tensor([1, 1, 1], dtype=ms.int32)
    z = Tensor([1, 1, 1], dtype=ms.int32)
    d = mutable(3)
    net = Net()
    ret = net(x, y, z, d)
    assert np.all(ret.asnumpy() == np.array((6, 6, 6)))
    os.environ['MS_DEV_ENABLE_REMOTE_MEMORY'] = '1'
