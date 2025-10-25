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
"""Elementwise operator test factory and utilities.

This module provides `ElementwiseOpsFactory` to build elementwise operator
tests with unified forward/grad comparison across backends.
"""
import mindspore as ms
from typing import Callable
from tests.st.ops.share._internal.meta import OpsFactory
from tests.st.ops.share._internal.utils import OpSampleInput, make_tensor
from tests.st.ops.share._op_info.op_info import OpInfo

class ElementwiseOpsFactory(OpsFactory):
    """Factory for elementwise ops testcases.

    Provides helpers to generate sample inputs of various shapes and run
    forward and gradient comparisons.
    """
    def __init__(
            self,
            *,
            op: Callable = None,
            ref: Callable = None,
            op_info: OpInfo = None,
            op_input=None,
            op_name=None,
            sample_inputs_func=None,
            **kwargs,
    ):
        super().__init__(
            op=op,
            ref=ref,
            op_info=op_info,
            op_input=op_input,
            op_args=(),
            op_kwargs={},
            op_name=op.__name__ if op is not None else "ElementwiseOp",
            sample_inputs_func=sample_inputs_func,
            **kwargs,
        )

    def test_elementwise_op_nd(
            self,
            shapes: list[tuple[int, ...]],
            dtype,
    ):
        '''Run elementwise op tests with 0-D to N-D tensors.
        Args:
            shapes: A list of shapes for the input tensors.
            dtype: The dtype of the input tensors.
        Returns:
            None
        '''
        no_grad_dtypes = [ms.bool_, ms.int8, ms.int16, ms.int32, ms.int64, ms.uint8, ms.uint16, ms.uint32, ms.uint64]
        def elementwise_op_nd_sample_inputs_func():
            sample_inputs = []
            for shape in shapes:
                sample_inputs.append(OpSampleInput(
                    op_input=make_tensor(shape, dtype),
                    op_args=self.op_args,
                    op_kwargs=self.op_kwargs,
                    op_name=self.op_name,
                ))
            return sample_inputs

        self.update_sample_inputs(elementwise_op_nd_sample_inputs_func)
        self.forward_cmp()
        if dtype not in no_grad_dtypes:
            self.grad_cmp()
