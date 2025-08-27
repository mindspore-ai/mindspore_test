# Copyright 2024 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark

import numpy as np
import mindspore as ms
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, mint
from mindspore.common import dtype as mstype
import torch
import pytest


# pylint: disable=W0235


def _ms_access_real_imag(x, accessor: str):
    if accessor == "func":
        return mint.real(x), mint.imag(x)
    return x.real(), x.imag()


def _torch_access_real_imag(x, accessor: str):
    if accessor == "func":
        return torch.real(x), torch.imag(x)
    return x.real, x.imag


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    """Compare if two numpy arrays are equal within tolerance"""
    if not np.allclose(data_expected, data_me, rtol, rtol, equal_nan=equal_nan):
        _count_unequal_element(data_expected, data_me, rtol, atol)
    else:
        assert np.array(data_expected).shape == np.array(data_me).shape


def _count_unequal_element(data_expected, data_me, rtol, atol):
    """Count the number of unequal elements"""
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_me) * rtol)
    nan_diff = np.not_equal(np.isnan(data_expected), np.isnan(data_me))
    inf_diff = np.not_equal(np.isinf(data_expected), np.isinf(data_me))
    neginf_diff = np.not_equal(np.isneginf(data_expected), np.isneginf(data_me))
    greater = greater + nan_diff + inf_diff + neginf_diff
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, \
        "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}". \
            format(data_expected[greater], data_me[greater], error[greater])


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("accessor", ["func", "attr"])  # func: mint.real/mint.imag, attr: x.real/x.imag
def test_real_imag_compare_torch(accessor):
    """
    Feature: Real/Imag ascend kernel compare with torch (function and tensor property)
    Description: Compare the real and imag view results between MindSpore and PyTorch via two access paths
    Expectation: outputs are the same as torch within tolerance
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

    def run_unified_test(real_data, imag_data, dtype_ms, dtype_torch, rtol, atol, _name):
        # MindSpore input and output
        real_ms = Tensor(real_data, dtype_ms)
        imag_ms = Tensor(imag_data, dtype_ms)
        x_ms = ops.Complex()(real_ms, imag_ms)
        ms_real, ms_imag = _ms_access_real_imag(x_ms, accessor)
        ms_real = ms_real.asnumpy()
        ms_imag = ms_imag.asnumpy()

        # PyTorch input and output
        real_torch = torch.tensor(real_data, dtype=dtype_torch)
        imag_torch = torch.tensor(imag_data, dtype=dtype_torch)
        x_torch = torch.complex(real_torch, imag_torch)
        torch_real, torch_imag = _torch_access_real_imag(x_torch, accessor)
        torch_real = torch_real.detach().numpy()
        torch_imag = torch_imag.detach().numpy()

        # NaN consistency
        if np.any(np.isnan(real_data)) or np.any(np.isnan(imag_data)):
            assert np.array_equal(np.isnan(ms_real), np.isnan(torch_real))
            assert np.array_equal(np.isnan(ms_imag), np.isnan(torch_imag))

        # Inf consistency
        if np.any(np.isinf(real_data)) or np.any(np.isinf(imag_data)):
            assert np.array_equal(np.isinf(ms_real), np.isinf(torch_real))
            assert np.array_equal(np.isinf(ms_imag), np.isinf(torch_imag))

        # Finite value comparison
        finite_mask_real = np.isfinite(ms_real)
        finite_mask_imag = np.isfinite(ms_imag)
        if np.any(finite_mask_real):
            allclose_nparray(torch_real[finite_mask_real], ms_real[finite_mask_real], rtol, atol)
        if np.any(finite_mask_imag):
            allclose_nparray(torch_imag[finite_mask_imag], ms_imag[finite_mask_imag], rtol, atol)

    # float32 complex numbers
    real_data = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
    imag_data = [7.7, 8.8, 9.9, 10.0, 11.1, 12.2]
    run_unified_test(real_data, imag_data, mstype.float32, torch.float32, 1e-5, 1e-5, "float32 complex")

    # float64 complex numbers
    run_unified_test(real_data, imag_data, mstype.float64, torch.float64, 1e-10, 1e-10, "float64 complex")

    # Random data
    np.random.seed(42)
    batch_size, channels, height, width = 2, 4, 8, 8
    real_data = np.random.randn(batch_size, channels, height, width).astype(np.float32)
    imag_data = np.random.randn(batch_size, channels, height, width).astype(np.float32)
    run_unified_test(real_data, imag_data, mstype.float32, torch.float32, 1e-5, 1e-5, "random data")

    # Zero values
    real_data = [0.0, 0.0, 0.0]
    imag_data = [0.0, 0.0, 0.0]
    run_unified_test(real_data, imag_data, mstype.float32, torch.float32, 1e-5, 1e-5, "zero values")

    # NaN values
    real_data = [np.nan, 1.0, np.nan]
    imag_data = [2.0, np.nan, np.nan]
    run_unified_test(real_data, imag_data, mstype.float32, torch.float32, 1e-5, 1e-5, "NaN values")

    # Infinity values
    real_data = [np.inf, -np.inf, 1.0]
    imag_data = [2.0, np.inf, -np.inf]
    run_unified_test(real_data, imag_data, mstype.float32, torch.float32, 1e-5, 1e-5, "infinity values")

    # Mixed special values
    real_data = [np.nan, np.inf, -np.inf, 0.0, 1.0]
    imag_data = [np.inf, np.nan, 0.0, -np.inf, 2.0]
    run_unified_test(real_data, imag_data, mstype.float32, torch.float32, 1e-5, 1e-5, "mixed special values")


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_real_imag_view_exceptions():
    """
    Feature: RealImagView ascend kernel exception handling
    Description: Test exception scenarios including TypeError and ValueError
    Expectation: appropriate exceptions are raised for invalid inputs
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

    def test_exception(exception_type, data=None, dtype=None, operation_name=None, test_type="type_error"):
        """Helper function for testing exceptions, supports TypeError and ValueError"""
        with pytest.raises(exception_type) as exc_info:
            view_op = mint.real if operation_name == "real_view" else mint.imag
            # TypeError test: non-complex type input
            input_tensor = Tensor(data, dtype)
            view_op(input_tensor)
            ms.runtime.synchronize()

        assert "complex64 or complex128" in str(exc_info.value)

    # Test case 1: TypeError - non-complex type input
    # Test float32 type
    test_exception(TypeError, [1.0, 2.0, 3.0], mstype.float32, "imag_view", "type_error")

    # Test float64 type
    test_exception(TypeError, [1.0, 2.0, 3.0], mstype.float64, "imag_view", "type_error")


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_real_imag_view_with_operations():
    """
    Feature: RealImagView ascend kernel with subsequent operations
    Description: Test that non-contiguous tensors from real_view and imag_view can be used in subsequent operations
    Expectation: view operations return tensors that work correctly with relu, groupnorm, etc.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

    # Create complex input data
    batch_size, channels, height, width = 2, 4, 8, 8
    real_data = np.random.randn(batch_size, channels, height, width).astype(np.float32)
    imag_data = np.random.randn(batch_size, channels, height, width).astype(np.float32)

    # MindSpore network
    class MSComplexProcessingNet(nn.Cell):
        def __init__(self):
            super(MSComplexProcessingNet, self).__init__()
            self.real_view = mint.real
            self.imag_view = mint.imag
            self.relu = ops.ReLU()
            self.sigmoid = ops.Sigmoid()
            self.tanh = ops.Tanh()
            self.add = ops.Add()
            self.mul = ops.Mul()

        def construct(self, complex_input):
            # Separate real and imaginary parts
            real_part = self.real_view(complex_input)
            imag_part = self.imag_view(complex_input)

            # Verify tensor continuity
            assert not real_part.is_contiguous(), "real_view should return non-contiguous tensor"
            assert not imag_part.is_contiguous(), "imag_view should return non-contiguous tensor"

            # Apply various operations to real part
            real_relu = self.relu(real_part)
            real_sigmoid = self.sigmoid(real_part)
            real_tanh = self.tanh(real_part)

            # Apply various operations to imag part
            imag_relu = self.relu(imag_part)
            imag_sigmoid = self.sigmoid(imag_part)
            imag_tanh = self.tanh(imag_part)

            # Mathematical operations
            real_plus_imag = self.add(real_part, imag_part)
            real_mul_imag = self.mul(real_part, imag_part)

            return (real_relu, real_sigmoid, real_tanh,
                    imag_relu, imag_sigmoid, imag_tanh,
                    real_plus_imag, real_mul_imag)

    # PyTorch network
    class TorchComplexProcessingNet(torch.nn.Module):
        def __init__(self):
            super(TorchComplexProcessingNet, self).__init__()
            self.relu = torch.nn.ReLU()
            self.sigmoid = torch.nn.Sigmoid()
            self.tanh = torch.nn.Tanh()

        def forward(self, complex_input):
            # Separate real and imaginary parts
            real_part = torch.real(complex_input)
            imag_part = torch.imag(complex_input)

            # Apply various operations to real part
            real_relu = self.relu(real_part)
            real_sigmoid = self.sigmoid(real_part)
            real_tanh = self.tanh(real_part)

            # Apply various operations to imag part
            imag_relu = self.relu(imag_part)
            imag_sigmoid = self.sigmoid(imag_part)
            imag_tanh = self.tanh(imag_part)

            # Mathematical operations
            real_plus_imag = real_part + imag_part
            real_mul_imag = real_part * imag_part

            return (real_relu, real_sigmoid, real_tanh,
                    imag_relu, imag_sigmoid, imag_tanh,
                    real_plus_imag, real_mul_imag)

    # Initialize networks
    ms_net = MSComplexProcessingNet()
    torch_net = TorchComplexProcessingNet()

    # MindSpore input and output
    real_ms = Tensor(real_data, mstype.float32)
    imag_ms = Tensor(imag_data, mstype.float32)
    complex_input_ms = ops.Complex()(real_ms, imag_ms)
    ms_output = ms_net(complex_input_ms)

    # PyTorch input and output
    real_torch = torch.tensor(real_data, dtype=torch.float32)
    imag_torch = torch.tensor(imag_data, dtype=torch.float32)
    complex_input_torch = torch.complex(real_torch, imag_torch)
    torch_output = torch_net(complex_input_torch)

    # Compare results
    rtol, atol = 1e-5, 1e-5

    # Unpack outputs
    (ms_real_relu, ms_real_sigmoid, ms_real_tanh,
     ms_imag_relu, ms_imag_sigmoid, ms_imag_tanh,
     ms_real_plus_imag, ms_real_mul_imag) = ms_output

    (torch_real_relu, torch_real_sigmoid, torch_real_tanh,
     torch_imag_relu, torch_imag_sigmoid, torch_imag_tanh,
     torch_real_plus_imag, torch_real_mul_imag) = torch_output

    # Verify shape and data type
    assert ms_real_relu.shape == torch_real_relu.shape
    assert ms_real_relu.dtype == mstype.float32
    assert torch_real_relu.dtype == torch.float32

    # Compare ReLU results
    allclose_nparray(torch_real_relu.detach().numpy(), ms_real_relu.asnumpy(), rtol, atol)
    allclose_nparray(torch_imag_relu.detach().numpy(), ms_imag_relu.asnumpy(), rtol, atol)

    # Compare Sigmoid results
    allclose_nparray(torch_real_sigmoid.detach().numpy(), ms_real_sigmoid.asnumpy(), rtol, atol)
    allclose_nparray(torch_imag_sigmoid.detach().numpy(), ms_imag_sigmoid.asnumpy(), rtol, atol)

    # Compare Tanh results
    allclose_nparray(torch_real_tanh.detach().numpy(), ms_real_tanh.asnumpy(), rtol, atol)
    allclose_nparray(torch_imag_tanh.detach().numpy(), ms_imag_tanh.asnumpy(), rtol, atol)

    # Compare mathematical operation results
    allclose_nparray(torch_real_plus_imag.detach().numpy(), ms_real_plus_imag.asnumpy(), rtol, atol)
    allclose_nparray(torch_real_mul_imag.detach().numpy(), ms_real_mul_imag.asnumpy(), rtol, atol)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_real_imag_view_with_normalization():
    """
    Feature: RealImagView ascend kernel with normalization operations
    Description: Test that non-contiguous tensors from real_view and imag_view can be used in normalization operations
    Expectation: normalization operations work correctly with view operation results
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

    # Create complex input data
    batch_size, channels, height, width = 2, 4, 8, 8
    real_data = np.random.randn(batch_size, channels, height, width).astype(np.float32)
    imag_data = np.random.randn(batch_size, channels, height, width).astype(np.float32)

    # MindSpore network
    class MSNormalizationNet(nn.Cell):
        def __init__(self, channels, height, width):
            super(MSNormalizationNet, self).__init__()
            self.real_view = mint.real
            self.imag_view = mint.imag
            self.layer_norm = mint.nn.LayerNorm([height, width])
            self.group_norm = mint.nn.GroupNorm(num_groups=2, num_channels=channels)

        def construct(self, complex_input):
            # Separate real and imaginary parts
            real_part = self.real_view(complex_input)
            imag_part = self.imag_view(complex_input)

            # Verify tensor continuity
            assert not real_part.is_contiguous(), "real_view should return non-contiguous tensor"
            assert not imag_part.is_contiguous(), "imag_view should return non-contiguous tensor"

            # Apply various normalizations to real part
            real_layer_norm = self.layer_norm(real_part)
            real_group_norm = self.group_norm(real_part)

            # Apply various normalizations to imag part
            imag_layer_norm = self.layer_norm(imag_part)
            imag_group_norm = self.group_norm(imag_part)

            return (real_layer_norm, real_group_norm,
                    imag_layer_norm, imag_group_norm)

    # PyTorch network
    class TorchNormalizationNet(torch.nn.Module):
        def __init__(self, channels, height, width):
            super(TorchNormalizationNet, self).__init__()
            self.layer_norm = torch.nn.LayerNorm([height, width])
            self.group_norm = torch.nn.GroupNorm(num_groups=2, num_channels=channels)

        def forward(self, complex_input):
            # Separate real and imaginary parts
            real_part = torch.real(complex_input)
            imag_part = torch.imag(complex_input)

            # Apply various normalizations to real part
            real_layer_norm = self.layer_norm(real_part)
            real_group_norm = self.group_norm(real_part)

            # Apply various normalizations to imag part
            imag_layer_norm = self.layer_norm(imag_part)
            imag_group_norm = self.group_norm(imag_part)

            return (real_layer_norm, real_group_norm,
                    imag_layer_norm, imag_group_norm)

    # Initialize networks
    ms_net = MSNormalizationNet(channels, height, width)
    torch_net = TorchNormalizationNet(channels, height, width)

    # MindSpore input and output
    real_ms = Tensor(real_data, mstype.float32)
    imag_ms = Tensor(imag_data, mstype.float32)
    complex_input_ms = ops.Complex()(real_ms, imag_ms)
    ms_output = ms_net(complex_input_ms)

    # PyTorch input and output
    real_torch = torch.tensor(real_data, dtype=torch.float32)
    imag_torch = torch.tensor(imag_data, dtype=torch.float32)
    complex_input_torch = torch.complex(real_torch, imag_torch)
    torch_output = torch_net(complex_input_torch)

    # Compare results
    rtol, atol = 1e-5, 1e-5

    # Unpack outputs
    (ms_real_layer_norm, ms_real_group_norm,
     ms_imag_layer_norm, ms_imag_group_norm) = ms_output

    (torch_real_layer_norm, torch_real_group_norm,
     torch_imag_layer_norm, torch_imag_group_norm) = torch_output

    # Verify shape and data type
    assert ms_real_layer_norm.shape == torch_real_layer_norm.shape
    assert ms_real_layer_norm.dtype == mstype.float32
    assert torch_real_layer_norm.dtype == torch.float32

    # Compare LayerNorm results
    allclose_nparray(torch_real_layer_norm.detach().numpy(), ms_real_layer_norm.asnumpy(), rtol, atol)
    allclose_nparray(torch_imag_layer_norm.detach().numpy(), ms_imag_layer_norm.asnumpy(), rtol, atol)

    # Compare GroupNorm results
    allclose_nparray(torch_real_group_norm.detach().numpy(), ms_real_group_norm.asnumpy(), rtol, atol)
    allclose_nparray(torch_imag_group_norm.detach().numpy(), ms_imag_group_norm.asnumpy(), rtol, atol)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_real_imag_view_backward():
    """
    Feature: RealImagView ascend kernel backward propagation
    Description: Test gradient computation and backward propagation for real and imag view operations
    Expectation: gradients are computed correctly and match PyTorch results
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

    # Create complex input data
    batch_size, channels, height, width = 2, 3, 4, 4
    real_data = np.random.randn(batch_size, channels, height, width).astype(np.float32)
    imag_data = np.random.randn(batch_size, channels, height, width).astype(np.float32)

    # MindSpore network
    class MSBackwardNet(nn.Cell):
        def __init__(self):
            super(MSBackwardNet, self).__init__()
            self.real_view = mint.real
            self.imag_view = mint.imag
            self.relu = ops.ReLU()
            self.sum = ops.ReduceSum()

        def construct(self, complex_input):
            # Separate real and imaginary parts
            real_part = self.real_view(complex_input)
            imag_part = self.imag_view(complex_input)

            # Apply activation functions
            real_relu = self.relu(real_part)
            imag_relu = self.relu(imag_part)

            # Calculate loss (sum of real and imaginary parts)
            loss = self.sum(real_relu) + self.sum(imag_relu)
            return loss

    # PyTorch network
    class TorchBackwardNet(torch.nn.Module):
        def __init__(self):
            super(TorchBackwardNet, self).__init__()

        def forward(self, complex_input):
            # Separate real and imaginary parts
            real_part = torch.real(complex_input)
            imag_part = torch.imag(complex_input)

            # Apply activation functions
            real_relu = torch.relu(real_part)
            imag_relu = torch.relu(imag_part)

            # Calculate loss (sum of real and imaginary parts)
            loss = torch.sum(real_relu) + torch.sum(imag_relu)
            return loss

    # Initialize networks
    ms_net = MSBackwardNet()
    torch_net = TorchBackwardNet()

    # MindSpore input
    real_ms = Tensor(real_data, mstype.float32)
    imag_ms = Tensor(imag_data, mstype.float32)
    complex_input_ms = ops.Complex()(real_ms, imag_ms)
    complex_input_ms.requires_grad = True

    # PyTorch input
    real_torch = torch.tensor(real_data, dtype=torch.float32)
    imag_torch = torch.tensor(imag_data, dtype=torch.float32)
    complex_input_torch = torch.complex(real_torch, imag_torch)
    complex_input_torch.requires_grad = True

    # Get torch grad
    torch_loss = torch_net(complex_input_torch)
    torch_loss.backward()
    torch_grad = complex_input_torch.grad

    # Get mindspore grad
    ms_grad_fn = ms.value_and_grad(ms_net, grad_position=0)
    ms_loss, ms_grad = ms_grad_fn(complex_input_ms)

    # Compare loss values
    rtol, atol = 1e-5, 1e-5
    allclose_nparray(torch_loss.detach().numpy(), ms_loss.asnumpy(), rtol, atol)

    # Compare gradient values
    allclose_nparray(torch_grad.detach().numpy(), ms_grad.asnumpy(), rtol, atol)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_real_view_non_complex_input_no_error():
    """
    Feature: RealView ascend kernel with non-complex input
    Description: real on non-complex tensor should not raise and keep dtype unchanged
    Expectation: no exception, output equals input and dtype unchanged
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

    for dt in (mstype.float16, mstype.float32, mstype.float64):
        data = np.array([1.25, -2.5, 3.0], dtype={
            mstype.float16: np.float16,
            mstype.float32: np.float32,
            mstype.float64: np.float64,
        }[dt])
        x = Tensor(data, dt)
        y = mint.real(x)
        assert y.dtype == dt
        np.testing.assert_allclose(y.asnumpy(), data, rtol=1e-5, atol=1e-5)

