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
- ReductionOpsFactory: constructs reduction operator testcases and enables
  forward/gradient comparisons across backends.
"""
import torch
from tests.st.ops.share._internal.meta import OpsFactory
from tests.st.ops.share._op_info.op_info import OpInfo

class ReductionOpsFactory(OpsFactory):
    """Factory for reduction ops testcases.

    Extends the common factory with net class tweaking and reduction-specific
    sample input builders.
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
        # Ensure pylint knows _douts is defined in this class.
        self._douts = None

    def grad_pytorch_impl(self):
        """Compute gradients using the PyTorch reference.

        Args:
            None

        Returns:
            list: Gradients for the first input tensor.

        Note:
            Overrides the base implementation. If the input dtype is not a
            floating or complex dtype, cast to torch.float32 for computation
            and convert the gradient back to the original dtype.
        """
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
        """Compute gradients with PyTorch for dynamic-shape cases.

        Args:
            None

        Returns:
            list: Gradients for dynamic-shape scenarios.

        Note:
            Overrides the base dynamic-shape implementation. If the input
            dtype is not a floating or complex dtype, compute gradients in
            torch.float32 and convert back to the original dtype.
        """
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
