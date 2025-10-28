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
- BinaryOpsFactory: utilities to compare MindSpore binary ops with Benchmark
  references.
- Static and dynamic-shape parity checks, with optional gradient comparisons.
"""
import torch
import mindspore as ms
from tests.st.ops.share._internal.meta import OpsFactory, OpCommonGradNetAllInput
from tests.st.ops.share._op_info.op_info import OpInfo
from tqdm import tqdm


class BinaryOpsFactory(OpsFactory):
    """Factory for testing binary operations.

    It wires up sample inputs, reference functions, and gradient networks to
    run value and gradient parity checks between MindSpore and Benchmark.
    """
    def __init__(
            self,
            op_info: OpInfo,
            **kwargs,
    ):
        super().__init__(
            op_info,
            **kwargs,
        )
        self.update_op_net_class(op_grad_net_class=OpCommonGradNetAllInput)
        # Ensure pylint knows _douts is defined in this class.
        self._douts = None

    def grad_pytorch_impl(self):
        """Compute gradients using the PyTorch reference.

        Args:
            None

        Returns:
            list: One or two gradients for the input tensors.

        Note:
            Use this function while op is Add, Adds, etc.
        """
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
        """Compute gradients with PyTorch for dynamic-shape cases.

        Args:
            None

        Returns:
            list: One or two gradients for the input tensors.

        Note:
            Use this function while op is Add, Adds, etc.
        """
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

    def test_binary_op_reference(
        self,
        *,
        grad_cmp=False
    ):
        """Run reference parity tests against Benchmark.

        Args:
            grad_cmp: When True, restrict to floating dtypes and compare
                first-order gradients in addition to forward results.
        """
        try:
            print(f"\nop_name: {self.op_name}, mode:{self._context_mode}, test_binary_op_reference...")
            if grad_cmp:
                self.supported_dtypes = tuple(d for d in self.supported_dtypes if d.is_floating_point)
            for dtype in tqdm(self.supported_dtypes):
                for sample_input in self.op_sample_inputs_func(self.op_info, dtype, device=self._device):
                    if grad_cmp:
                        self.compare_with_torch(sample_inputs=sample_input, grad_cmp=True)
                    else:
                        self.compare_with_torch(sample_inputs=sample_input)
        except Exception as e:
            print(f"\ntest_binary_op_reference failed:"
                  f"\nop_name: {self.op_name}"
                  f"\nmode: {self._context_mode}"
                  f"\ndtype: {dtype}"
                  f"\n{sample_input.summary(True)}")
            raise e


    def test_binary_op_dynamic(
        self,
        *,
        dynamic_mode='dynamic_shape',
        grad_cmp=False,
    ):
        """Run dynamic-shape tests against Benchmark.

        Args:
            dynamic_mode: Dynamic mode identifier, e.g. 'dynamic_shape'.
            grad_cmp: When True, also compare first-order gradients.
        """
        try:
            print(f"\nop_name: {self.op_name}, dynamic_mode={dynamic_mode}, test_binary_op_dynamic...")
            self.op_sample_inputs_func = self.op_info.op_dynamic_inputs_func
            sample_input = self.op_sample_inputs_func(self.op_info,
                                                      dtype=ms.float32,
                                                      device=self._device,
                                                      dynamic_mode=dynamic_mode)
            if grad_cmp:
                self.compare_with_torch_dynamic(sample_inputs=sample_input, grad_cmp=True)
            else:
                self.compare_with_torch_dynamic(sample_inputs=sample_input)
        except Exception as e:
            print(f"\test_binary_op_dynamic failed:"
                  f"\nop_name: {self.op_name}"
                  f"\ndynamic_mode: {dynamic_mode}")
            raise e
