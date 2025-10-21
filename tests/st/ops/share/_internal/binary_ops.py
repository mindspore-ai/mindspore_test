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
from tests.st.ops.share._internal.meta import OpsFactory, OpCommonGradNetAllInput
from tests.st.ops.share._internal.utils import OpSampleInput, make_tensor
from tests.st.ops.share._op_info.op_info import OpInfo, dtypes_integral
from typing import Callable
from functools import partial


class BinaryOpsFactory(OpsFactory):
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
            op_name=op.__name__ if op is not None else "BinaryOp",
            sample_inputs_func=sample_inputs_func,
            **kwargs,
        )
        self.update_op_net_class(op_grad_net_class=OpCommonGradNetAllInput)
        # Ensure pylint knows _douts is defined in this class.
        self._douts = None

    def grad_pytorch_impl(self):
        '''
        Compute the gradient of the binary op with the PyTorch implementation.
        Args:
            None
        Note:
            Use this function while op is Add, Adds, etc.
        Returns:
            A list of gradients (one or two gradient) for the input tensors.
        '''
        torch_douts = self._generate_random_dout(return_torch_douts=True)

        torch_fn = self.ref
        grads = []

        for idx, sample_input in enumerate(self._sample_inputs):
            if self._inplace_op:
                sample_input = sample_input.copy()
            sample_input = sample_input.astorch()
            op_input, op_args, op_kwargs = sample_input.op_input, sample_input.op_args, sample_input.op_kwargs

            if isinstance(op_input, torch.Tensor):
                op_input.requires_grad = True # input
            if isinstance(sample_input.op_args[0], torch.Tensor):
                sample_input.op_args[0].requires_grad = True # other

            outi = torch_fn(op_input, *op_args, **op_kwargs)
            outi.backward(gradient=torch_douts[idx])

            if isinstance(op_input, torch.Tensor) and isinstance(sample_input.op_args[0], torch.Tensor):
                grads.append((op_input.grad.detach(), sample_input.op_args[0].grad.detach()))
            elif isinstance(op_input, torch.Tensor):
                grads.append((op_input.grad.detach(),))
            elif isinstance(sample_input.op_args[0], torch.Tensor):
                grads.append((sample_input.op_args[0].grad.detach(),))
            else:
                raise ValueError(f"BinaryOpsFactory.grad_pytorch_impl suppose one or two input tensors, "
                                 f"but got {type(op_input)} and {type(sample_input.op_args[0])}")
        return grads

    def grad_pytorch_dynamic_shape_impl(self):
        '''
        Compute the gradient of the binary op with the PyTorch implementation for dynamic shape.
        Args:
            None
        Returns:
            A list of gradients (one or two gradient) for the input tensors.
        Note:
            Use this function while op is Add, Adds, etc.
        '''
        torch_fn = self.ref
        grads = []

        for sample_input in self._sample_inputs[1:]:
            if self._inplace_op:
                sample_input = sample_input.copy()
            sample_input = sample_input.astorch()
            op_input, op_args, op_kwargs = sample_input.op_input, sample_input.op_args, sample_input.op_kwargs

            if isinstance(op_input, torch.Tensor):
                op_input.requires_grad = True # input
            if isinstance(sample_input.op_args[0], torch.Tensor):
                sample_input.op_args[0].requires_grad = True # other

            outi = torch_fn(op_input, *op_args, **op_kwargs)
            outi_grad = torch.ones_like(outi)
            outi.backward(gradient=outi_grad)

            if isinstance(op_input, torch.Tensor) and isinstance(sample_input.op_args[0], torch.Tensor):
                grads.append((op_input.grad.detach(), sample_input.op_args[0].grad.detach()))
            elif isinstance(op_input, torch.Tensor):
                grads.append((op_input.grad.detach(),))
            elif isinstance(sample_input.op_args[0], torch.Tensor):
                grads.append((sample_input.op_args[0].grad.detach(),))
            else:
                raise ValueError(f"BinaryOpsFactory.grad_pytorch_dynamic_shape_impl suppose one or two input tensors, "
                                 f"but got {type(op_input)} and {type(sample_input.op_args[0])}")
        return grads

    def test_binary_op_nd_same_dtype(
            self,
            *,
            op_kwargs={},
            dtypes: list = None,
            backend: str = None,
            disable_op_info_dtypes: list = None,
            broad_cast=False,
    ):

        def binary_op_nd_sample_inputs_func(dtype, broad_cast, op_kwargs):
            shapes = [
                (2,),
                (2, 3),
                (2, 3, 2),
                (2, 3, 2, 2),
                (2, 3, 2, 2, 3),
                (2, 3, 2, 2, 3, 2),
                (2, 3, 2, 2, 3, 2, 2),
                (2, 3, 2, 2, 3, 2, 2, 2),
                (2, 0, 3)
            ]

            broad_cast_shapes = [
                (1,),
                (2, 1),
                (2, 1, 2),
                (2, 3, 1, 2),
                (2, 3, 1, 2, 3),
                (2, 3, 2, 1, 3, 2),
                (2, 3, 2, 1, 3, 2, 2),
                (2, 3, 2, 2, 1, 2, 2, 2),
                (2, 0, 1)
            ]

            sample_inputs = []
            for idx, shape in enumerate(shapes):
                input_shape = shape
                other_shape = shape if not broad_cast else broad_cast_shapes[idx]
                x = make_tensor(input_shape, dtype)
                y = make_tensor(other_shape, dtype)
                sample_inputs.append(OpSampleInput(
                    op_input=x,
                    op_args=(y,),
                    op_kwargs=op_kwargs,
                    op_name=self.op_name,
                ))
            return sample_inputs


        if not dtypes:
            dtypes = self.op_info.get_dtypes(backend)
        if disable_op_info_dtypes:
            dtypes = [dtype for dtype in dtypes if dtype not in disable_op_info_dtypes]

        is_differentiable = self.op_info.is_differentiable if self.op_info is not None else True

        for dtype in dtypes:
            sample_inputs_func = partial(binary_op_nd_sample_inputs_func, dtype, broad_cast, self.op_kwargs)
            self.update_sample_inputs(sample_inputs_func)
            self.forward_cmp()
            if is_differentiable and dtype not in dtypes_integral:
                # generate new douts for each dtype.
                self._douts = None
                self.grad_cmp()
