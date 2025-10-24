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
import torch
import mindspore as ms
from mindspore import nn
from typing import Callable, Union
from tests.st.ops.share._internal.meta import OpsFactory
from tests.st.ops.share._internal.utils import OpSampleInput, make_tensor
from tests.st.ops.share._op_info.op_info import OpInfo

class ReductionOpNetNoKwargs(nn.Cell):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def construct(self, *op_args):
        return self.op(*op_args[:-1], dtype=op_args[-1])

class ReductionOpsFactory(OpsFactory):
    def __init__(
            self,
            *,
            op: Callable = None,
            ref: Callable = None,
            op_info: OpInfo = None,
            op_input=None,
            op_args=(),
            op_kwargs={},
            op_name=None,
            sample_inputs_func=None,
            **kwargs,
    ):
        super().__init__(
            op=op,
            ref=ref,
            op_info=op_info,
            op_input=op_input,
            op_args=op_args,
            op_kwargs=op_kwargs,
            op_name=op.__name__ if op is not None else "ReductionOp",
            sample_inputs_func=sample_inputs_func,
            **kwargs,
        )
        self.update_op_net_class(op_net_class_no_kwargs=ReductionOpNetNoKwargs)
        # Ensure pylint knows _douts is defined in this class.
        self._douts = None

    def test_reduction_op_nd(
            self,
            dtype,
            op_params: Union[tuple[dict, ...], list[dict, ...]]
    ):
        '''
        Test the reduction op with 0-D to N-D input tensors.
        Args:
            op_params: A tuple or list of parameters for the reduction op.
        Returns:
            None
        '''
        no_grad_dtypes = [ms.bool, ms.int8, ms.int16, ms.int32, ms.int64, ms.uint8, ms.uint16, ms.uint32, ms.uint64]
        def reduction_op_nd_sample_inputs_func():
            sample_inputs = []
            for op_param in op_params:
                sample_inputs.append(OpSampleInput(
                    op_input=make_tensor(op_param['shape'], dtype),
                    op_args=(op_param['dim'], op_param['keepdim']),
                    op_kwargs=dict(dtype=op_param['dtype']),
                    op_name=self.op_name,
                ))
            return sample_inputs

        self.update_sample_inputs(reduction_op_nd_sample_inputs_func)
        self.forward_cmp()
        if dtype not in no_grad_dtypes:
            self.grad_cmp()

    def grad_pytorch_impl(self):
        '''
        Compute the gradient of the op with the PyTorch implementation.
        Args:
            None
        Returns:
            A list of gradients for the op_input.
        Note:
            This is a override function of the OpFactory.grad_pytorch_impl.
            If the dtype of the op_input is not a floating point or complex dtype,
            use torch.float32 to compute the gradient, then convert the gradient to the original dtype finally.
        '''
        torch_douts = self._generate_random_dout(return_torch_douts=True)

        torch_fn = self.ref
        grads = []

        for idx, sample_input in enumerate(self._sample_inputs):
            if self._inplace_op:
                sample_input = sample_input.copy()
            sample_input = sample_input.astorch()

            op_input, op_args, op_kwargs = sample_input.op_input, sample_input.op_args, sample_input.op_kwargs

            original_dtype = None
            if not op_input.dtype.is_floating_point and not op_input.dtype.is_complex:
                original_dtype = op_input.dtype
                op_input = op_input.to(torch.float32)
                torch_douts = [d.to(torch.float32) for d in torch_douts]

            op_input.requires_grad = True

            outi = torch_fn(op_input, *op_args, **op_kwargs)
            outi.backward(gradient=torch_douts[idx])

            gradi = op_input.grad.detach()
            gradi = gradi.to(original_dtype) if original_dtype is not None else gradi

            grads.append(gradi)

        return grads

    def grad_pytorch_dynamic_shape_impl(self):
        '''
        Compute the gradient of the op with the PyTorch implementation for dynamic shape.
        Args:
            None
        Note:
            This is a override function of the OpFactory.grad_pytorch_dynamic_shape_impl.
            If the dtype of the op_input is not a floating point or complex dtype,
            use torch.float32 to compute the gradient, then convert the gradient to the original dtype finally.
        Returns:
            A list of gradients for the dynamic shape.
        '''
        torch_fn = self.ref
        grads = []

        for sample_input in self._sample_inputs[1:]:
            if self._inplace_op:
                sample_input = sample_input.copy()
            sample_input = sample_input.astorch()
            op_input, op_args, op_kwargs = sample_input.op_input, sample_input.op_args, sample_input.op_kwargs

            original_dtype = None
            if not op_input.dtype.is_floating_point and not op_input.dtype.is_complex:
                original_dtype = op_input.dtype
                op_input = op_input.to(torch.float32)
            op_input.requires_grad = True

            outi = torch_fn(op_input, *op_args, **op_kwargs)
            outi_grad = torch.ones_like(outi)
            outi.backward(gradient=outi_grad)

            gradi = op_input.grad.detach()
            gradi = gradi.to(original_dtype) if original_dtype is not None else gradi

            grads.append(gradi)

        return grads
