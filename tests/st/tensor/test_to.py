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
import time
import pytest
import numpy as np
from tests.mark_utils import arg_mark
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, value_and_grad, jit
from mindspore.ops.auto_generate.gen_ops_prim import to_device_op, to_dtype_op


def test_to_base():
    """
    Feature: to_device_op
    Description: Test to_device_op
    Expectation: success
    """

    dtypes = [ms.float16, ms.float32, ms.float64, ms.int32, ms.int64]
    devices = ["Ascend", "CPU"]

    for device_type in devices:
        x = Tensor(1.0, ms.float32)
        y = x.to(device_type)
        assert device_type in y.device

    for dtype in dtypes:
        x = Tensor(1.0, ms.float32)
        y = x.to(dtype)
        assert dtype == y.dtype

    for device_type in devices:
        for dtype in dtypes:
            x = Tensor(1.0, ms.float32)
            y = x.to(device_type, dtype)
            assert device_type in y.device
            assert dtype == y.dtype

    for device_type in devices:
        for dtype in dtypes:
            for non_blocking in [True, False]:
                for copy in [True, False]:
                    x = Tensor(1.0, ms.float32)
                    # Test overload function api parser with kwargs.
                    y = x.to(device=device_type, dtype=dtype, non_blocking=non_blocking, copy=copy)
                    assert device_type in y.device
                    assert dtype == y.dtype
                    if copy:
                        assert y.data_ptr() != x.data_ptr()

    for device_type in devices:
        for dtype in dtypes:
            for non_blocking in [True, False]:
                for copy in [True, False]:
                    x = Tensor(1.0, ms.float32)
                    # Test overload function api parser.
                    y = x.to(device_type, dtype, non_blocking, copy)
                    assert device_type in y.device
                    assert dtype == y.dtype
                    if copy:
                        assert y.data_ptr() != x.data_ptr()

    for device_type in devices:
        for dtype in dtypes:
            for non_blocking in [True, False]:
                for copy in [True, False]:
                    x = Tensor(1.0, ms.float32)
                    other = Tensor(2.0, dtype).move_to(device_type)
                    y = x.to(other, non_blocking, copy)
                    assert device_type in y.device
                    assert dtype == y.dtype
                    if copy:
                        assert y.data_ptr() != x.data_ptr()


def test_to_jit():
    """
    Feature: Tensor.to
    Description: Test Tensor.to with jit
    Expectation: success
    """

    @jit
    def tensor_to_1(x, device, non_blocking):
        return x.to(device, non_blocking=non_blocking)

    @jit
    def tensor_to_2(x, dtype):
        return x.to(dtype)

    @jit
    def tensor_to_3(x, device, dtype, non_blocking, copy):
        return x.to(device, dtype, non_blocking, copy)

    @jit
    def tensor_to_4(x, dtype, non_blocking, copy):
        return x.to(dtype, non_blocking, copy)

    @jit
    def tensor_to_5(x, other, non_blocking, copy):
        return x.to(other, non_blocking, copy)

    dtypes = [ms.float16, ms.float32, ms.float64, ms.int32, ms.int64]
    devices = ["Ascend", "CPU"]

    for device_type in devices:
        x = Tensor(1.0, ms.float32)
        for non_blocking in [True, False]:
            y = tensor_to_1(x, device_type, non_blocking)
            assert y.asnumpy() == 1.0

    for dtype in dtypes:
        x = Tensor(1.0, ms.float32)
        y = tensor_to_2(x, dtype)
        assert dtype == y.dtype
        assert y.asnumpy() == 1.0

    for device_type in devices:
        for dtype in dtypes:
            x = Tensor(1.0, ms.float32)
            y = tensor_to_3(x, device_type, dtype, False, False)
            assert dtype == y.dtype
            assert y.asnumpy() == 1.0

    for dtype in dtypes:
        x = Tensor(1.0, ms.float32)
        y = tensor_to_4(x, dtype, False, False)
        assert dtype == y.dtype
        assert y.asnumpy() == 1.0

    for device_type in devices:
        for dtype in dtypes:
            x = Tensor(1.0, ms.float32)
            y = Tensor(2.0, dtype).move_to(device_type)
            z = tensor_to_5(x, y, False, False)
            assert dtype == y.dtype
            assert z.asnumpy() == 1.0


def test_to_device_grad():
    """
    Feature: to_device_op
    Description: Test to_device_op grad
    Expectation: success
    """

    class Net(nn.Cell):
        def construct(self, x, y, z):
            w = to_device_op(x, "Ascend")
            return w * y * z

    x = Tensor([1, 2], ms.float32)
    y = Tensor([-2, 3], ms.float32)
    z = Tensor([0, 3], ms.float32)

    net = Net()
    grad_fn = value_and_grad(net, grad_position=0)
    output, inputs_grad = grad_fn(x, y, z)
    assert (output.asnumpy() == [-0.0, 18.0]).all()
    assert (inputs_grad.asnumpy() == [-0.0, 9.0]).all()


def test_tensor_api_to_device_grad():
    """
    Feature: Tensor.to(device)
    Description: Test Tensor.to(device) grad
    Expectation: success
    """

    class Net(nn.Cell):
        def construct(self, x, y, z):
            w = x.to("Ascend")
            return w * y * z

    x = Tensor([1, 2], ms.float32)
    y = Tensor([-2, 3], ms.float32)
    z = Tensor([0, 3], ms.float32)

    net = Net()
    grad_fn = value_and_grad(net, grad_position=0)
    output, inputs_grad = grad_fn(x, y, z)
    assert (output.asnumpy() == [-0.0, 18.0]).all()
    assert (inputs_grad.asnumpy() == [-0.0, 9.0]).all()


def test_to_dtype_grad():
    """
    Feature: Tensor.to(dtype)
    Description: Test Tensor.to(dtype) grad
    Expectation: success
    """
    class Net(nn.Cell):
        def construct(self, x, y, z):
            x = to_dtype_op(x, ms.float16)
            y = to_dtype_op(y, ms.float16)
            z = to_dtype_op(z, ms.float16)
            return x * y * z

    net = Net()
    grad_fn = value_and_grad(net, grad_position=0)

    x = Tensor([1, 2], ms.float32)
    y = Tensor([-2, 3], ms.float32)
    z = Tensor([0, 3], ms.float32)
    output, inputs_grad = grad_fn(x, y, z)
    assert output.dtype == ms.float16
    assert inputs_grad.dtype == ms.float32
    assert (output.asnumpy() == [-0.0, 18.0]).all()
    assert (inputs_grad.asnumpy() == [-0.0, 9.0]).all()


def test_tensor_api_to_dtype_grad():
    """
    Feature: Tensor.to(dtype)
    Description: Test Tensor.to(dtype) grad
    Expectation: success
    """
    class Net(nn.Cell):
        def construct(self, x, y, z):
            x = x.to(ms.float16)
            y = y.to(ms.float16)
            z = z.to(ms.float16)
            return x * y * z

    net = Net()
    grad_fn = value_and_grad(net, grad_position=0)

    x = Tensor([1, 2], ms.float32)
    y = Tensor([-2, 3], ms.float32)
    z = Tensor([0, 3], ms.float32)
    output, inputs_grad = grad_fn(x, y, z)
    assert output.dtype == ms.float16
    assert inputs_grad.dtype == ms.float32
    assert (output.asnumpy() == [-0.0, 18.0]).all()
    assert (inputs_grad.asnumpy() == [-0.0, 9.0]).all()


def test_tensor_numpy_non_blocking():
    """
    Feature: Tensor._numpy_non_blocking
    Description: Test convert to numpy non_blocking.
    Expectation: success
    """
    np_data = np.ones((10, 10))
    x = Tensor(np_data).to("Ascend").sin()
    output = x.to("CPU")._numpy_non_blocking() # pylint:disable=protected-access
    ms.runtime.synchronize()
    assert np.allclose(output, np.sin(np_data))


def test_tensor_to_with_numpy_perf():
    """
    Feature: Tensor to performance
    Description: Test Tensor.to and Tensor._numpy_non_blocking performance.
    Expectation: success
    """
    def non_blocking_perf():
        ms.runtime.synchronize()
        np_data = np.ones((1000, 1000))
        x = Tensor(np_data).to("Ascend").sin()
        start = time.time()
        for _ in range(1000):
            x.to("CPU", non_blocking=True)._numpy_non_blocking() # pylint:disable=protected-access
        ms.runtime.synchronize()
        end = time.time()
        return end - start

    def blocking_perf():
        ms.runtime.synchronize()
        start = time.time()
        np_data = np.ones((1000, 1000))
        x = Tensor(np_data).to("Ascend").sin()
        for _ in range(1000):
            x.asnumpy()
        ms.runtime.synchronize()
        end = time.time()
        return end - start

    assert non_blocking_perf() < blocking_perf()


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_tensor_to(mode):
    """
    Feature: tensor.to
    Description: Test tensor.to overload
    Expectation: success
    """
    ms.set_context(mode=mode)
    ms.set_context(jit_level="O0")

    test_to_base()
    test_to_jit()

    test_to_device_grad()
    test_to_dtype_grad()

    test_tensor_api_to_device_grad()
    test_tensor_api_to_dtype_grad()

    test_tensor_numpy_non_blocking()
