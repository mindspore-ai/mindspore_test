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
# pylint: disable=R1705
import torch
import numpy as np
import mindspore as ms
from mindspore import nn
from typing import Callable, Optional, final
from tests.st.utils.test_utils import single_golden_compare, double_golden_compare
from tests.st.ops.share._internal.utils import OpSampleInput, make_tensor, ms_asnumpy
from tests.st.ops.share._op_info.op_info import OpInfo

class OpsCommonNet(nn.Cell):
    '''
    default forward op net class.
    Use this class while you don't want to use a Specialized OpNet.
    '''
    def __init__(self, op):
        super().__init__()
        self.op = op

    def construct(self, op_input, *op_args, **op_kwargs):
        return self.op(op_input, *op_args, **op_kwargs)


class OpsCommonNetNoKwargs(nn.Cell):
    '''
    Used for get gradient of the op with graph mode, because op_kwargs must convert to op_args while sens_param is True.
    So we need to use this class to construct the op net without kwargs.
    '''
    def __init__(self, op):
        super().__init__()
        self.op = op

    def construct(self, *op_args):
        return self.op(*op_args)


class OpCommonGradNetFirstInput(nn.Cell):
    '''
    Used for get gradient of the op with first input.
    Before use this class, make sure all op_kwargs are converted to op_args.
    Use OpSampleInput.convert_to_args() to covert all op_kwargs to op_args and append the dout to the op_args.
    '''
    def __init__(self, network, *, sens_param=True):
        super().__init__()
        self.network = network
        self.grad = ms.ops.GradOperation(sens_param=sens_param)(self.network)

    def construct(self, *op_args):
        return self.grad(*op_args)


class OpCommonGradNetAllInput(nn.Cell):
    '''
    Used for get gradient of the op with all inputs.
    Before use this class, make sure all op_kwargs are converted to op_args.
    Use OpSampleInput.convert_to_args() to covert all op_kwargs to op_args and append the dout to the op_args.
    '''
    def __init__(self, network, *, sens_param=True):
        super().__init__()
        self.network = network
        self.grad = ms.ops.GradOperation(get_all=True, sens_param=sens_param)(self.network)

    def construct(self, *op_args):
        return self.grad(*op_args)


class OpsFactory():
    def __init__(
            self,
            *,
            op: Callable = None,
            ref: Callable = None,
            op_info: OpInfo = None,
            op_input=None,
            op_args: Optional[tuple] = tuple(),
            op_kwargs: Optional[dict] = dict(),
            op_name: Optional[str] = None,
            sample_inputs_func=None,
            **kwargs,
    ):
        self.op = op                                 # mindspore interface
        self.ref = ref                               # reference implementation, such as pytorch, tensorflow, numpy
        self.op_info = op_info                       # op info, such as add_ext, sum, etc.
        self.op_input = op_input                     # input for op
        self.op_args = op_args                       # args for op
        self.op_kwargs = op_kwargs                   # kwargs for op
        self.op_name = op_name                       # name of the op
        self.sample_inputs_func = sample_inputs_func # function to generate sample inputs

        if self.op_name is None:
            self.op_name = self.op.__name__ if self.op is not None else "UnknownOp"

        if self.op_info is not None:
            self.op = self.op_info.op
            self.ref = self.op_info.ref
            self.op_name = self.op_info.name

        if self.sample_inputs_func is None:
            self.sample_inputs_func = self._default_sample_inputs_func

        self._sample_inputs = self.sample_inputs_func()
        self._douts = None
        self._op_net_class = OpsCommonNet
        self._op_net_class_no_kwargs = OpsCommonNetNoKwargs
        self._op_grad_net_class = OpCommonGradNetFirstInput

        # if op is inplace op, set _inplace_op to True. then all tensors will be copied before forward and grad.
        self._inplace_op = kwargs['inplace_op'] if 'inplace_op' in kwargs else False

        # if op does not support float16 on certain backend of benchmark,
        # such as sum of torch gpu can't support float16.
        # set convert_half_to_float to True, then the float16 will be converted to float32 for benchmark calculation,
        # and convert back to float16 for comparison. op of torch gpu don't support float16 usually.
        self._convert_half_to_float = ms.context.get_context('device_target').lower() == 'gpu'
        if 'convert_half_to_float' in kwargs:
            self._convert_half_to_float = kwargs['convert_half_to_float']
        # if op need to compare extra uint dtypes, set convert_extra_uint to True.
        # then the extra uint dtypes will be converted to int64 in torch implementation for comparison.
        # op of torch don't support extra uint dtypes, so set convert_extra_uint to True by default.
        self._convert_extra_uint = kwargs['convert_extra_uint'] if 'convert_extra_uint' in kwargs else True

    @final
    def update_op_net_class(
            self,
            *,
            op_net_class=None,
            op_net_class_no_kwargs=None,
            op_grad_net_class=None
    ):
        '''
        Update the op net class and op grad net class.
        Args:
            op_net_class: The op net class to use for the op.
            op_net_class_no_kwargs: The op net class to use for the op without kwargs,
                                    used for dynamic shape and gradient.
            op_grad_net_class: The op grad net class to use for the op.
        Returns:
            None
        '''
        self._op_net_class = op_net_class if op_net_class is not None else self._op_net_class
        self._op_net_class_no_kwargs = op_net_class_no_kwargs \
            if op_net_class_no_kwargs is not None else self._op_net_class_no_kwargs
        self._op_grad_net_class = op_grad_net_class if op_grad_net_class is not None else self._op_grad_net_class

    @final
    def update_sample_inputs(
            self,
            sample_inputs_func,
    ):
        '''
        Update the sample inputs for the op.
        Args:
            sample_inputs_func: The function to generate the sample inputs.
        Returns:
            None
        '''
        self.sample_inputs_func = sample_inputs_func
        self._sample_inputs = self.sample_inputs_func()

    @final
    def _default_sample_inputs_func(self):
        '''
        Generate the sample inputs for the op by default.
        Returns:
            A list of OpSampleInput objects.
        '''
        return [OpSampleInput(
            self.op_input,
            self.op_args,
            self.op_kwargs,
            self.op_name,
        )]

    def _generate_random_dout(self, return_torch_douts=False):
        '''
        Generate the random dout for the op.
        Args:
            return_torch_douts: Whether to return the torch douts.
        Returns:
            A list of random douts or None.
        '''
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
    def set_context_mode(
            self,
            *,
            mode=None,
            jit_level=None
    ):
        '''
        Set the context mode for the op.
        Args:
            mode: The mode to use for the op.
            jit_level: The JIT level to use for the op.
        Returns:
            None
        '''
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
        if jit_level is not None:
            ms.context.set_context(jit_level=jit_level)

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
        '''
        Assert the equality of the actual and expect outputs.
        Args:
            actual: The actual output of the op.
            expect: The expect output of the op.
            rtol: The relative tolerance for the comparison.
            atol: The absolute tolerance for the comparison.
            compare_method: The method to use for the comparison.
            ksize: The kernel size for the comparison.
            op_type: The type of the op.
            secend_expect: The second expect output of the op.
            convert_extra_uint: Whether to compare the extra uint dtype.
        Note:
            You can override this function to implement the comparison logic with other implementations.
        Returns:
            None
        '''
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

            if actual.dtype in (ms.float16, torch.float16, np.float16):
                loss = 1e-3
            elif actual.dtype in (
                    ms.float32, ms.complex64, torch.float32, torch.complex64,
                    np.float32, np.complex64):
                loss = 1e-4
            elif actual.dtype in (
                    ms.float64, ms.complex128, torch.float64, torch.complex128,
                    np.float64, np.complex128):
                loss = 1e-5
            elif actual.dtype in (ms.bfloat16, torch.bfloat16):
                loss = 4e-3
            else:
                loss = 0

            rtol = loss if rtol is None else rtol
            atol = loss if atol is None else atol

            actual = convert_tensor_to_nparray(actual)
            expect = convert_tensor_to_nparray(expect)

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

        if self._convert_extra_uint:
            expect = convert_mindspore_extra_uint_dtype_to_int64(expect)
            actual = convert_mindspore_extra_uint_dtype_to_int64(actual)

        if self._convert_half_to_float:
            expect, actual = convert_torch_float_to_half(expect, actual)

        if compare_method == 'default_golden':
            default_golden_compare(expect, actual, rtol, atol)
        elif compare_method == 'single_golden':
            single_golden_compare(expect, actual, ksize, op_type)
        elif compare_method == 'double_golden':
            double_golden_compare(expect, secend_expect, actual, op_type)
        else:
            raise ValueError(f"Invalid compare_method: {compare_method}, expected: 'default_golden', 'single_golden', \
                              'double_golden'.")

    def forward_mindspore_impl(
            self,
            *args,
            **kwargs
    ):
        '''
        Forward the op with the MindSpore implementation.
        Args:
            *args: The positional arguments for 'forward_mindspore_impl'.
            **kwargs: The keyword arguments for 'forward_mindspore_impl'.
        Note:
            You can override this function to implement the forward logic with other implementations.
        Returns:
            A single tensor or a list of tensors
        '''
        op_net = self._op_net_class(self.op)
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
        '''
        Forward the op with the PyTorch implementation.
        Args:
            *args: The positional arguments for 'forward_pytorch_impl'.
            **kwargs: The keyword arguments for 'forward_pytorch_impl'.
        Note:
            You can override this function to implement the forward logic with other implementations.
        Returns:
            A single tensor or a list of tensors.
        '''
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
        '''
        Forward the op with the TensorFlow implementation.
        Args:
            *args: The positional arguments for 'forward_tensorflow_impl'.
            **kwargs: The keyword arguments for 'forward_tensorflow_impl'.
        Note:
            You can override this function to implement the forward logic with other implementations.
        Returns:
            A single tensor or a list of tensors
        '''
        raise NotImplementedError

    def forward_numpy_impl(
            self,
            *args,
            **kwargs
    ):
        '''
        Forward the op with the NumPy implementation.
        Args:
            *args: The positional arguments for 'forward_numpy_impl'.
            **kwargs: The keyword arguments for 'forward_numpy_impl'.
        Note:
            You can override this function to implement the forward logic with other implementations.
        Returns:
            A single tensor or a list of tensors
        '''
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
        '''
        Compute the gradient of the op with the MindSpore implementation.
        Args:
            *args: The positional arguments for 'grad_mindspore_impl'.
            **kwargs: The keyword arguments for 'grad_mindspore_impl'.
        Note:
            You should override this function if you don't want to only get the gradient for first input tensor.
        Returns:
            A list of gradients for op_input.
        '''
        self._generate_random_dout()

        net = self._op_net_class_no_kwargs(self.op)
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
        '''
        Compute the gradient of the op with the PyTorch implementation.
        Args:
            *args: The positional arguments for 'grad_pytorch_impl'.
            **kwargs: The keyword arguments for 'grad_pytorch_impl'.
        Note:
            You should override this function if you don't want to only get the gradient for first input tensor.
        Returns:
            A list of gradients for op_input.
        '''
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
        '''
        Compute the gradient of the op with the TensorFlow implementation.
        Args:
            *args: The positional arguments for 'grad_tensorflow_impl'.
            **kwargs: The keyword arguments for 'grad_tensorflow_impl'.
        Note:
            You should override this function if you don't want to only get the gradient for first input tensor.
        Returns:
            A list of gradients for op_input.
        '''
        raise NotImplementedError

    def grad_numpy_impl(
            self,
            *args,
            **kwargs
    ):
        '''
        Compute the gradient of the op with the NumPy implementation.
        Args:
            *args: The positional arguments for 'grad_numpy_impl'.
            **kwargs: The keyword arguments for 'grad_numpy_impl'.
        Note:
            You should override this function if you don't want to only get the gradient for first input tensor.
        Returns:
            A list of gradients for op_input.
        '''
        raise NotImplementedError


    def forward_mindspore_dynamic_shape_impl(
            self,
            *args,
            **kwargs
    ):
        '''
        Forward the op with the MindSpore implementation for dynamic shape.
        Args:
            *args: The positional arguments for 'forward_mindspore_dynamic_shape_impl'.
            **kwargs: The keyword arguments for 'forward_mindspore_dynamic_shape_impl'.
        Note:
            You should override this function if you want to implement the forward logic with other implementations.
        Returns:
            A list of outputs for the dynamic shape.
        '''
        op_net = self._op_net_class_no_kwargs(self.op)
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
        '''
        Forward the op with the PyTorch implementation for dynamic shape.
        Args:
            *args: The positional arguments for 'forward_pytorch_dynamic_shape_impl'.
            **kwargs: The keyword arguments for 'forward_pytorch_dynamic_shape_impl'.
        Note:
            You should override this function if you want to implement the forward logic with other implementations.
        Returns:
            A list of outputs for the dynamic shape.
        '''
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
        '''
        Compute the gradient of the op with the MindSpore implementation for dynamic shape.
        Args:
            *args: The positional arguments for 'grad_mindspore_dynamic_shape_impl'.
            **kwargs: The keyword arguments for 'grad_mindspore_dynamic_shape_impl'.
        Note:
            You should override this function if you want to implement the gradient logic with other implementations.
        Returns:
            A list of gradients for the dynamic shape.
        '''
        net = self._op_net_class_no_kwargs(self.op)
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
        '''
        Compute the gradient of the op with the PyTorch implementation for dynamic shape.
        Args:
            *args: The positional arguments for 'grad_pytorch_dynamic_shape_impl'.
            **kwargs: The keyword arguments for 'grad_pytorch_dynamic_shape_impl'.
        Note:
            You should override this function if you want to implement the gradient logic with other implementations.
        Returns:
            A list of gradients for the dynamic shape.
        '''
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

    def forward_cmp(
            self,
            *args,
            rtol=None,
            atol=None,
            benchmark='torch',
            **kwargs,
    ):
        '''
        Compare the output of the op with the output of the reference implementation.
        Args:
            rtol: The relative tolerance for the comparison.
            atol: The absolute tolerance for the comparison.
            benchmark: The benchmark to use for the comparison.
        Note:
            The 'forward_cmp' function in OpFactory should be overridden to implement
            the comparison logic while the output of op is not a single tensor.
        Returns:
            None
        '''
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
        '''
        Compare the gradient of the op with the gradient of the reference implementation.
        Args:
            rtol: The relative tolerance for the comparison.
            atol: The absolute tolerance for the comparison.
            benchmark: The benchmark to use for the comparison.
        Note:
            You should override this function if you don't want to only get the gradient for first input tensor.
        Returns:
            None
        '''
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
        '''
        Compare the output of the op with the output of the reference implementation for dynamic shape.
        Args:
            rtol: The relative tolerance for the comparison.
            atol: The absolute tolerance for the comparison.
            benchmark: The benchmark to use for the comparison.
        Note:
            You should override this function if you want to implement the forward logic with other implementations.
        Returns:
            None
        '''
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
        '''
        Compare the gradient of the op with the gradient of the reference implementation for dynamic shape.
        Args:
            rtol: The relative tolerance for the comparison.
            atol: The absolute tolerance for the comparison.
            benchmark: The benchmark to use for the comparison.
        Note:
            You should override this function if you want to implement the gradient logic with other implementations.
        Returns:
            None
        '''
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
