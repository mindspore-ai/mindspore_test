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
"""Utility helpers for operation testing.

This module provides:
- OpsFactory: a base test factory handling context/device, sample inputs,
  and comparisons.
- Helper networks: forward/grad nets such as OpsCommonNet, OpsCommonNetNoKwargs,
  OpCommonGradNetFirstInput, and OpCommonGradNetAllInput.
- Comparison routines: static and dynamic-shape forward/grad parity checks
  against reference backends.
"""
# pylint: disable=R1705
import torch
import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore._c_expression import MSContext
from mindspore.common.dtype import _dtype_to_nptype
from typing import Optional, Union, List, final
from tests.st.utils.test_utils import single_golden_compare, double_golden_compare, OpTypes
from tests.st.ops.share._internal.utils import OpSampleInput, make_tensor, ms_asnumpy
from tests.st.ops.share._op_info.op_info import OpInfo
from tests.st.ops.share._op_info.op_common import get_default_loss, dtypes_extra_uint
class OpsCommonNet(nn.Cell):
    """Default forward op net wrapper.

    Use this class when a specialized op net is not needed.
    """
    def __init__(self, op):
        super().__init__()
        self.op = op

    def construct(self, op_input, *op_args, **op_kwargs):
        return self.op(op_input, *op_args, **op_kwargs)


class OpsCommonNetNoKwargs(nn.Cell):
    """Forward op net wrapper without kwargs for grad/dynamic.

    Used in graph mode where kwargs must be converted to args while
    sens_param=True.
    """
    def __init__(self, op):
        super().__init__()
        self.op = op

    def construct(self, *op_args):
        return self.op(*op_args)


class OpCommonGradNetFirstInput(nn.Cell):
    """Gradient network for the first input.

    Before use, ensure op_kwargs are converted to op_args using
    OpSampleInput.convert_to_args() and append dout to op_args.
    """
    def __init__(self, network, *, sens_param=True):
        super().__init__()
        self.network = network
        self.grad = ms.ops.GradOperation(sens_param=sens_param)(self.network)

    def construct(self, *op_args):
        return self.grad(*op_args)


class OpCommonGradNetAllInput(nn.Cell):
    """Gradient network for all inputs.

    Before use, ensure op_kwargs are converted to op_args using
    OpSampleInput.convert_to_args() and append dout to op_args.
    """
    def __init__(self, network, *, sens_param=True):
        super().__init__()
        self.network = network
        self.grad = ms.ops.GradOperation(get_all=True, sens_param=sens_param)(self.network)

    def construct(self, *op_args):
        return self.grad(*op_args)


class OpsFactory():
    """Base test factory for operators.

    Manages device/context, builds sample inputs, forwards through MindSpore
    and references, and performs value/gradient comparisons.
    """
    def __init__(
            self,
            op_info: OpInfo,
            **kwargs,
    ):
        self.op_info = op_info
        # inner params
        self._douts = None
        self._device = None
        self._context_mode = 'pynative'
        self._op_net_class = OpsCommonNet
        self._op_net_class_no_kwargs = OpsCommonNetNoKwargs
        self._op_grad_net_class = OpCommonGradNetFirstInput

        self._parse_op_info(self.op_info)

    @final
    def _parse_op_info(self, op_info: OpInfo):
        """Populate factory fields from `OpInfo` and current device context.

        Args:
            op_info: Operator metadata including op callable, reference, dtypes,
                sample input builder, compare method, etc.
        """
        self.op = op_info.op
        self.op_func_grad = op_info.op_func_grad
        self.ref = op_info.ref
        self.op_name = op_info.name
        self.op_sample_inputs_func = op_info.op_sample_inputs_func
        self._sample_inputs = None

        # get supported dtypes for the op with entire environment.
        device = ms.context.get_context('device_target').lower()
        if device == 'ascend':
            if MSContext.get_instance().get_ascend_soc_version() == 'ascend910b':
                self.supported_dtypes = op_info.dtypes_ascend910b
            else:
                self.supported_dtypes = op_info.dtypes_ascend
        elif device == 'cpu':
            self.supported_dtypes = op_info.dtypes_cpu
        elif device == 'gpu':
            self.supported_dtypes = op_info.dtypes_gpu
        else:
            raise ValueError(f"Invalid device: {device}, expected: 'ascend', 'cpu', 'gpu'.")

        self._device = device
        self._inplace_op = getattr(op_info, 'is_inplace_op', False)
        # op of torch don't support extra uint dtypes, so set convert_extra_uint to True if mindspore supports them.
        self._convert_extra_uint = bool(set(self.supported_dtypes) & set(dtypes_extra_uint))

        self._convert_half_to_float = getattr(op_info, 'convert_half_to_float', False)
        if not self._convert_half_to_float:
            # if op does not support float16 on certain backend of benchmark,
            # such as sum of torch gpu can't support float16.
            # the float16 will be converted to float32 for benchmark calculation,
            # and convert back to float16 for comparison. op of torch gpu don't support float16 usually.
            self._convert_half_to_float = device == 'gpu'

        self._compare_method = op_info.compare_method
        self._default_golden_loss_func = op_info.default_golden_loss_func

    @final
    def _generate_random_dout(self, return_torch_douts=False):
        """Generate random dout tensors for the op.

        Args:
            return_torch_douts (bool): Whether to return PyTorch douts.

        Returns:
            list | None: Random douts or None when not requested.
        """
        if self._douts is None:
            ms_out = self.forward_mindspore_impl()
            self._douts = [make_tensor(outi.shape, outi.dtype, random_method='randn') for outi in ms_out]

        if return_torch_douts:
            torch_douts = [torch.tensor(ms_asnumpy(d)) for d in self._douts]
            if self._convert_half_to_float:
                torch_douts = [d.float() if d.dtype == torch.float16 else d for d in torch_douts]
            return torch_douts
        return None

    @final
    def update_op_net_class(
            self,
            *,
            op_net_class=None,
            op_net_class_no_kwargs=None,
            op_grad_net_class=None
    ):
        """Update forward/grad network wrappers used by the factory.

        Args:
            op_net_class: Net class for standard forward execution.
            op_net_class_no_kwargs: Net class without kwargs (dynamic/grad).
            op_grad_net_class: Net class for gradient computation.
        """
        self._op_net_class = op_net_class if op_net_class is not None else self._op_net_class
        self._op_net_class_no_kwargs = op_net_class_no_kwargs \
            if op_net_class_no_kwargs is not None else self._op_net_class_no_kwargs
        self._op_grad_net_class = op_grad_net_class if op_grad_net_class is not None else self._op_grad_net_class

    @final
    def update_sample_inputs(
            self,
            op_sample_inputs_func=None,
    ):
        """Update the sample input generator and refresh samples.

        Args:
            op_sample_inputs_func: Function that generates sample inputs.
        """
        if op_sample_inputs_func is not None:
            self.op_sample_inputs_func = op_sample_inputs_func
        self._sample_inputs = self.op_sample_inputs_func()

    @final
    def set_context_mode(
            self,
            *,
            mode=None
    ):
        """Set the execution context mode for the op.

        Args:
            mode: One of 'kbk', 'ge', 'pynative', or a MindSpore mode enum.
        """
        if mode is not None:
            if isinstance(mode, str):
                if mode.lower() == 'kbk':
                    ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')
                elif mode.lower() == 'ge':
                    ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O2')
                elif mode.lower() == 'pynative':
                    ms.context.set_context(mode=ms.PYNATIVE_MODE)
                else:
                    raise ValueError(f"Invalid mode: {mode}, expected: 'kbk', 'ge', 'pynative'.")
            else:
                ms.context.set_context(mode=mode)
            self._context_mode = mode

    @final
    def assert_equal(
            self,
            actual,
            expect,
            rtol=None,
            atol=None,
            *,
            compare_method='default_golden',
            ksize=None,
            op_type=None,
            secend_expect=None,
    ):
        """Assert equality within tolerances using configured comparison.

        Args:
            actual: Actual output.
            expect: Expected output.
            rtol: Relative tolerance.
            atol: Absolute tolerance.
            compare_method: 'default_golden' | 'single_golden' | 'double_golden'.
            ksize: Kernel size for certain comparisons.
            op_type: Operation type enum for golden comparisons.
            secend_expect: Second expected output (for double golden).

        Note:
            Override to plug in other comparison strategies if needed.
        """
        def _count_unequal_element(expect, actual, rtol, atol):
            assert expect.shape == actual.shape
            total_count = len(expect.flatten())
            error = np.abs(expect - actual)
            greater = np.greater(error, atol + np.abs(actual) * rtol)
            nan_diff = np.not_equal(np.isnan(expect), np.isnan(actual))
            inf_diff = np.not_equal(np.isinf(expect), np.isinf(actual))
            if expect.dtype in ('complex64', 'complex128'):
                greater = greater + nan_diff + inf_diff
            else:
                neginf_diff = np.not_equal(np.isneginf(expect), np.isneginf(actual))
                greater = greater + nan_diff + inf_diff + neginf_diff
            loss_count = np.count_nonzero(greater)
            assert (loss_count / total_count) < rtol, \
                "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}". \
                    format(expect[greater], actual[greater], error[greater])

        def allclose_nparray(expect, actual, rtol, atol, equal_nan=True):
            if not np.allclose(expect, actual, rtol, atol, equal_nan=equal_nan):
                _count_unequal_element(expect, actual, rtol, atol)
            else:
                assert np.array(expect).shape == np.array(actual).shape

        def default_golden_compare(expect, actual, rtol, atol):
            def convert_tensor_to_nparray(tensor):
                if isinstance(tensor, torch.Tensor):
                    return tensor.float().cpu().numpy() if tensor.dtype == torch.bfloat16 else tensor.cpu().numpy()
                if isinstance(tensor, ms.Tensor):
                    return ms_asnumpy(tensor)
                return tensor

            actual = convert_tensor_to_nparray(actual)
            expect = convert_tensor_to_nparray(expect)

            if self._convert_extra_uint:
                if actual.dtype in (map(_dtype_to_nptype, dtypes_extra_uint)) and expect.dtype == np.int64:
                    expect = expect.astype(actual.dtype)
                if expect.dtype in (map(_dtype_to_nptype, dtypes_extra_uint)) and actual.dtype == np.int64:
                    actual = actual.astype(expect.dtype)

            rtol = get_default_loss(actual.dtype) if rtol is None else rtol
            atol = get_default_loss(actual.dtype) if atol is None else atol

            allclose_nparray(expect, actual, rtol, atol)

        def convert_mindspore_extra_uint_dtype_to_int64(tensor):
            extra_uint_dtypes = [ms.uint16, ms.uint32, ms.uint64]
            if isinstance(tensor, ms.Tensor) and tensor.dtype in extra_uint_dtypes:
                return tensor.to(ms.int64)
            return tensor

        def convert_torch_float_to_half(x, y):
            if isinstance(x, ms.Tensor) and x.dtype == ms.float16:
                if isinstance(y, torch.Tensor) and y.dtype == torch.float32:
                    y = y.to(torch.float16)
            if isinstance(y, ms.Tensor) and y.dtype == ms.float16:
                if isinstance(x, torch.Tensor) and x.dtype == torch.float32:
                    x = x.to(torch.float16)
            return x, y

        if self._convert_extra_uint and compare_method != 'default_golden':
            expect = convert_mindspore_extra_uint_dtype_to_int64(expect)
            actual = convert_mindspore_extra_uint_dtype_to_int64(actual)

        if self._convert_half_to_float:
            expect, actual = convert_torch_float_to_half(expect, actual)

        if compare_method == 'default_golden':
            default_golden_compare(expect, actual, rtol, atol)
        elif compare_method == 'single_golden':
            assert single_golden_compare(expect, actual, ksize, op_type)
        elif compare_method == 'double_golden':
            assert double_golden_compare(expect, secend_expect, actual, op_type)
        else:
            raise ValueError(f"Invalid compare_method: {compare_method}, expected: 'default_golden', 'single_golden', \
                              'double_golden'.")

    def forward_mindspore_impl(
            self,
            *args,
            **kwargs
    ):
        """Run forward with the MindSpore implementation.

        Args:
            *args: Positional arguments (unused; present for API symmetry).
            **kwargs: Keyword arguments (unused; present for API symmetry).

        Returns:
            list: Outputs per sample input.
        """
        op_net = self.op if self._context_mode == 'pynative' else self._op_net_class(self.op)
        out = []

        for sample_input in self._sample_inputs:
            if self._inplace_op:
                sample_input = sample_input.copy()
            op_input, op_args, op_kwargs = sample_input.op_input, sample_input.op_args, sample_input.op_kwargs
            outi = op_net(op_input, *op_args, **op_kwargs)
            out.append(outi)

        return out

    def forward_pytorch_impl(
            self,
            *args,
            **kwargs
    ):
        """Run forward with the PyTorch reference implementation.

        Args:
            *args: Positional arguments (unused; present for API symmetry).
            **kwargs: Keyword arguments (unused; present for API symmetry).

        Returns:
            list: Outputs per sample input.
        """
        torch_fn = self.ref
        out = []

        for sample_input in self._sample_inputs:
            if self._inplace_op:
                sample_input = sample_input.copy()
            sample_input = sample_input.astorch(convert_half_to_float=self._convert_half_to_float,
                                                convert_extra_uint=self._convert_extra_uint)
            op_input, op_args, op_kwargs = sample_input.op_input, sample_input.op_args, sample_input.op_kwargs
            outi = torch_fn(op_input, *op_args, **op_kwargs)
            out.append(outi)

        return out

    def forward_tensorflow_impl(
            self,
            *args,
            **kwargs
    ):
        """Run forward with the TensorFlow reference implementation.

        Args:
            *args: Positional arguments (unused; present for API symmetry).
            **kwargs: Keyword arguments (unused; present for API symmetry).
        """
        raise NotImplementedError

    def forward_numpy_impl(
            self,
            *args,
            **kwargs
    ):
        """Run forward with the NumPy reference implementation.

        Args:
            *args: Positional arguments (unused; present for API symmetry).
            **kwargs: Keyword arguments (unused; present for API symmetry).

        Returns:
            list: Outputs per sample input.
        """
        np_fn = self.ref
        out = []

        for sample_input in self._sample_inputs:
            if self._inplace_op:
                sample_input = sample_input.copy()
            sample_input = sample_input.asnumpy()
            op_input, op_args, op_kwargs = sample_input.op_input, sample_input.op_args, sample_input.op_kwargs

            outi = np_fn(op_input, *op_args, **op_kwargs)
            out.append(outi)

        return out

    def grad_mindspore_impl(
            self,
            *args,
            **kwargs
    ):
        """Compute gradients with the MindSpore implementation.

        Args:
            *args: Positional arguments (unused; present for API symmetry).
            **kwargs: Keyword arguments (unused; present for API symmetry).

        Returns:
            list: Gradients per sample input.
        """
        self._douts = None
        self._generate_random_dout()

        net = self._op_net_class_no_kwargs(self.op_func_grad)
        grad_net = self._op_grad_net_class(net)
        grads = []

        for idx, sample_input in enumerate(self._sample_inputs):
            if self._inplace_op:
                sample_input = sample_input.copy()
            sample_input = sample_input.convert_to_args(append_dout=self._douts[idx])

            # After convert_to_args, op_input, op_args, op_kwargs and dout are all in op_args now.
            grad_outi = grad_net(*sample_input.op_args)
            grads.append(grad_outi)

        return grads

    def grad_pytorch_impl(
            self,
            *args,
            **kwargs
    ):
        """Compute gradients with the PyTorch reference implementation.

        Args:
            *args: Positional arguments (unused; present for API symmetry).
            **kwargs: Keyword arguments (unused; present for API symmetry).

        Returns:
            list: Gradients per sample input.
        """
        torch_douts = self._generate_random_dout(return_torch_douts=True)

        torch_fn = self.ref
        grads = []

        for idx, sample_input in enumerate(self._sample_inputs):
            if self._inplace_op:
                sample_input = sample_input.copy()
            sample_input = sample_input.astorch(convert_half_to_float=self._convert_half_to_float)
            op_input, op_args, op_kwargs = sample_input.op_input, sample_input.op_args, sample_input.op_kwargs
            op_input.requires_grad = True

            outi = torch_fn(op_input, *op_args, **op_kwargs)
            outi.backward(gradient=torch_douts[idx])
            grads.append(op_input.grad.detach())

        return grads

    def grad_tensorflow_impl(
            self,
            *args,
            **kwargs
    ):
        """Compute gradients with the TensorFlow reference implementation."""
        raise NotImplementedError

    def grad_numpy_impl(
            self,
            *args,
            **kwargs
    ):
        """Compute gradients with the NumPy reference implementation."""
        raise NotImplementedError


    def forward_mindspore_dynamic_shape_impl(
            self,
            *args,
            **kwargs
    ):
        """Run forward with MindSpore for dynamic-shape execution.

        Args:
            *args: Positional arguments (unused; present for API symmetry).
            **kwargs: Keyword arguments (unused; present for API symmetry).

        Returns:
            list: Outputs per dynamic-shape sample.
        """
        op_net = self._op_net_class_no_kwargs(self.op_func_grad)
        compile_input = self._sample_inputs[0]
        compile_input = compile_input.convert_to_args()
        op_net.set_inputs(*compile_input.op_args)
        out = []

        for sample_input in self._sample_inputs[1:]:
            if self._inplace_op:
                sample_input = sample_input.copy()

            sample_input = sample_input.convert_to_args()
            outi = op_net(*sample_input.op_args)
            out.append(outi)

        return out

    def forward_pytorch_dynamic_shape_impl(
            self,
            *args,
            **kwargs
    ):
        """Run forward with PyTorch for dynamic-shape execution.

        Args:
            *args: Positional arguments (unused; present for API symmetry).
            **kwargs: Keyword arguments (unused; present for API symmetry).

        Returns:
            list: Outputs per dynamic-shape sample.
        """
        torch_fn = self.ref
        out = []

        for sample_input in self._sample_inputs[1:]:
            if self._inplace_op:
                sample_input = sample_input.copy()
            sample_input = sample_input.astorch(convert_half_to_float=self._convert_half_to_float,
                                                convert_extra_uint=self._convert_extra_uint)
            op_input, op_args, op_kwargs = sample_input.op_input, sample_input.op_args, sample_input.op_kwargs
            outi = torch_fn(op_input, *op_args, **op_kwargs)
            out.append(outi)

        return out

    def grad_mindspore_dynamic_shape_impl(
            self,
            *args,
            **kwargs
    ):
        """Compute gradients with MindSpore for dynamic-shape execution.

        Args:
            *args: Positional arguments (unused; present for API symmetry).
            **kwargs: Keyword arguments (unused; present for API symmetry).

        Returns:
            list: Gradients per dynamic-shape sample.
        """
        net = self._op_net_class_no_kwargs(self.op_func_grad)
        grad_net = self._op_grad_net_class(net, sens_param=False)
        compile_sample_input = self._sample_inputs[0]
        compile_sample_input = compile_sample_input.convert_to_args()
        grad_net.set_inputs(*compile_sample_input.op_args)
        grads = []

        for sample_input in self._sample_inputs[1:]:
            if self._inplace_op:
                sample_input = sample_input.copy()
            sample_input = sample_input.convert_to_args()

            # After convert_to_args, op_input, op_args and op_kwargs are all in op_args now.
            grad_outi = grad_net(*sample_input.op_args)
            grads.append(grad_outi)

        return grads

    def grad_pytorch_dynamic_shape_impl(
            self,
            *args,
            **kwargs
    ):
        """Compute gradients with PyTorch for dynamic-shape execution.

        Args:
            *args: Positional arguments (unused; present for API symmetry).
            **kwargs: Keyword arguments (unused; present for API symmetry).

        Returns:
            list: Gradients per dynamic-shape sample.
        """
        torch_fn = self.ref
        grads = []

        for sample_input in self._sample_inputs[1:]:
            if self._inplace_op:
                sample_input = sample_input.copy()
            sample_input = sample_input.astorch(convert_half_to_float=self._convert_half_to_float)
            op_input, op_args, op_kwargs = sample_input.op_input, sample_input.op_args, sample_input.op_kwargs

            op_input.requires_grad = True

            outi = torch_fn(op_input, *op_args, **op_kwargs)
            outi_grad = torch.ones_like(outi)
            outi.backward(gradient=outi_grad)
            grads.append(op_input.grad.detach())

        return grads

    def compare_with_torch(
            self,
            *,
            sample_inputs: Union[List[OpSampleInput], OpSampleInput],
            grad_cmp: Optional[bool] = False,
            ksize: Optional[int] = 1, # ksize for elementwise op, set other value if you want
    ):
        """Compare MindSpore outputs/gradients with PyTorch on static shapes.

        Args:
            sample_inputs: Single or list of sample inputs.
            grad_cmp: When True and differentiable, compare gradients.
            ksize: Optional kernel size hint for comparison helpers.
        """
        self._sample_inputs = sample_inputs if isinstance(sample_inputs, list) else [sample_inputs]

        if grad_cmp and self.op_info.is_differentiable:
            ms_out = self.grad_mindspore_impl()
            pt_out = self.grad_pytorch_impl()
        else:
            ms_out = self.forward_mindspore_impl()
            pt_out = self.forward_pytorch_impl()

        for ms_outi, pt_outi in zip(ms_out, pt_out):
            if isinstance(ms_outi, (tuple, list)) and isinstance(pt_outi, (tuple, list)):
                # The output of the op maybe a tuple or list for some multi-output ops.
                for ms_outi_tensor, pt_outi_tensor in zip(ms_outi, pt_outi):
                    loss = self._default_golden_loss_func(ms_outi_tensor.dtype)
                    self.assert_equal(
                        ms_outi_tensor,
                        pt_outi_tensor,
                        rtol=loss,
                        atol=loss,
                        compare_method=self._compare_method,
                        ksize=ksize,
                        op_type=OpTypes.COMPUTE_FLOAT
                    )
            else:
                loss = self._default_golden_loss_func(ms_outi.dtype)
                self.assert_equal(
                    ms_outi,
                    pt_outi,
                    rtol=loss,
                    atol=loss,
                    compare_method=self._compare_method,
                    ksize=ksize,
                    op_type=OpTypes.COMPUTE_FLOAT
                )

    def compare_with_torch_dynamic(
            self,
            *,
            sample_inputs: Union[List[OpSampleInput], OpSampleInput],
            grad_cmp: Optional[bool] = False,
            ksize: Optional[int] = 1, # ksize for elementwise op, set other value if you want
    ):
        """Compare MindSpore with PyTorch under dynamic-shape execution.

        Args:
            sample_inputs: Single or list of sample inputs; first is for compile.
            grad_cmp: When True and differentiable, compare gradients.
            ksize: Optional kernel size hint for comparison helpers.
        """
        self._sample_inputs = sample_inputs if isinstance(sample_inputs, list) else [sample_inputs]

        if grad_cmp and self.op_info.is_differentiable:
            ms_out = self.grad_mindspore_dynamic_shape_impl()
            pt_out = self.grad_pytorch_dynamic_shape_impl()
        else:
            ms_out = self.forward_mindspore_dynamic_shape_impl()
            pt_out = self.forward_pytorch_dynamic_shape_impl()

        for ms_outi, pt_outi in zip(ms_out, pt_out):
            if isinstance(ms_outi, (tuple, list)) and isinstance(pt_outi, (tuple, list)):
                # The output of the op maybe a tuple or list for some multi-output ops.
                for ms_outi_tensor, pt_outi_tensor in zip(ms_outi, pt_outi):
                    self.assert_equal(ms_outi_tensor, pt_outi_tensor)
            else:
                self.assert_equal(ms_outi, pt_outi)

    def forward_cmp(
            self,
            *args,
            rtol=None,
            atol=None,
            benchmark='torch',
            **kwargs,
    ):
        """Compare MindSpore forward results with a reference implementation.

        Args:
            rtol: Relative tolerance.
            atol: Absolute tolerance.
            benchmark: 'torch' | 'numpy'.
        """
        ms_out = self.forward_mindspore_impl()
        if benchmark == 'torch':
            pt_out = self.forward_pytorch_impl()
        elif benchmark == 'numpy':
            pt_out = self.forward_numpy_impl()
        else:
            raise ValueError(f"Invalid benchmark: {benchmark}, expected: 'torch', 'numpy'.")

        for ms_outi, pt_outi in zip(ms_out, pt_out):
            if isinstance(ms_outi, (tuple, list)) and isinstance(pt_outi, (tuple, list)):
                # The output of the op maybe a tuple or list for some multi-output ops.
                for ms_outi_tensor, pt_outi_tensor in zip(ms_outi, pt_outi):
                    self.assert_equal(ms_outi_tensor, pt_outi_tensor, rtol, atol)
            else:
                self.assert_equal(ms_outi, pt_outi, rtol, atol)

    def grad_cmp(
            self,
            *args,
            rtol=None,
            atol=None,
            benchmark='torch',
            **kwargs,
    ):
        """Compare MindSpore gradients with a reference implementation.

        Args:
            rtol: Relative tolerance.
            atol: Absolute tolerance.
            benchmark: 'torch' | 'numpy'.
        """
        ms_grads = self.grad_mindspore_impl()
        if benchmark == 'torch':
            pt_grads = self.grad_pytorch_impl()
        elif benchmark == 'numpy':
            pt_grads = self.grad_numpy_impl()
        else:
            raise ValueError(f"Invalid benchmark: {benchmark}, expected: 'torch', 'numpy'.")

        for ms_gradi, pt_gradi in zip(ms_grads, pt_grads):
            if isinstance(ms_gradi, (tuple, list)) and isinstance(pt_gradi, (tuple, list)):
                # The gradient of the op maybe a tuple or list for some multi-tensor input ops.
                for ms_gradi_tensor, pt_gradi_tensor in zip(ms_gradi, pt_gradi):
                    self.assert_equal(ms_gradi_tensor, pt_gradi_tensor, rtol, atol)
            else:
                self.assert_equal(ms_gradi, pt_gradi, rtol, atol)

    def forward_dynamic_shape_cmp(
            self,
            *args,
            rtol=None,
            atol=None,
            benchmark='torch',
            **kwargs,
    ):
        """Compare forward results under dynamic-shape execution.

        Args:
            rtol: Relative tolerance.
            atol: Absolute tolerance.
            benchmark: 'torch'.
        """
        if self._context_mode == 'pynative':
            raise RuntimeError("Dynamic shape comparison is not supported in pynative mode.")

        ms_outs = self.forward_mindspore_dynamic_shape_impl()
        if benchmark == 'torch':
            pt_outs = self.forward_pytorch_dynamic_shape_impl()
        else:
            raise ValueError(f"Invalid benchmark: {benchmark}, expected: 'torch'.")
        for ms_outi, pt_outi in zip(ms_outs, pt_outs):
            if isinstance(ms_outi, (tuple, list)) and isinstance(pt_outi, (tuple, list)):
                # The output of the op with dynamic shape maybe a tuple or list for some multi-output ops.
                for ms_outi_tensor, pt_outi_tensor in zip(ms_outi, pt_outi):
                    self.assert_equal(ms_outi_tensor, pt_outi_tensor, rtol, atol)
            else:
                self.assert_equal(ms_outi, pt_outi, rtol, atol)

    def grad_dynamic_shape_cmp(
            self,
            *args,
            rtol=None,
            atol=None,
            benchmark='torch',
            **kwargs,
    ):
        """Compare gradients under dynamic-shape execution.

        Args:
            rtol: Relative tolerance.
            atol: Absolute tolerance.
            benchmark: 'torch'.
        """
        if self._context_mode == 'pynative':
            raise RuntimeError("Dynamic shape comparison is not supported in pynative mode.")

        ms_grads = self.grad_mindspore_dynamic_shape_impl()
        if benchmark == 'torch':
            pt_grads = self.grad_pytorch_dynamic_shape_impl()
        else:
            raise ValueError(f"Invalid benchmark: {benchmark}, expected: 'torch'.")
        for ms_gradi, pt_gradi in zip(ms_grads, pt_grads):
            if isinstance(ms_gradi, (tuple, list)) and isinstance(pt_gradi, (tuple, list)):
                # The gradient of the op maybe a tuple or list for some multi-tensor input ops.
                for ms_gradi_tensor, pt_gradi_tensor in zip(ms_gradi, pt_gradi):
                    self.assert_equal(ms_gradi_tensor, pt_gradi_tensor, rtol, atol)
            else:
                self.assert_equal(ms_gradi, pt_gradi, rtol, atol)
