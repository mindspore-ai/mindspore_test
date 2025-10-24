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
"""Add/Sub operator tests using MindSpore Mint APIs.

Provides `AddSubMintOpFactory` to exercise mixed dtype and scalar/tensor
combinations with forward/grad comparisons.
"""
import mindspore as ms
from mindspore import nn
from typing import Callable
from tests.st.ops.share._internal.binary_ops import BinaryOpsFactory
from tests.st.ops.share._internal.utils import make_tensor, OpSampleInput
from tests.st.ops.share._op_info.op_info import OpInfo

class AddSubMintNetNoKwargs(nn.Cell):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def construct(self, *op_args):
        return self.op(*op_args[:-1], alpha=op_args[-1])

class AddSubMintOpFactory(BinaryOpsFactory):
    """Factory for mint.add/mint.sub style tests.

    Generates diverse sample inputs and runs backend comparisons.
    """
    def __init__(
            self,
            *,
            op: Callable = None,
            ref: Callable = None,
            op_info: OpInfo = None,
            op_input=None,
            op_args=(),
            op_kwargs=None,
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
            op_kwargs=op_kwargs if op_kwargs is not None else {},
            op_name=op_name,
            sample_inputs_func=sample_inputs_func,
            **kwargs,
        )
        self.update_op_net_class(op_net_class_no_kwargs=AddSubMintNetNoKwargs)
        self.integer_dtypes = [ms.bool_, ms.int8, ms.int16, ms.int32, ms.int64,
                               ms.uint8, ms.uint16, ms.uint32, ms.uint64]
        self.extra_uint_dtypes = [ms.uint16, ms.uint32, ms.uint64]

    def get_tensor_by_dtype(self, shape, dtype):
        if dtype in self.integer_dtypes:
            return make_tensor(shape, dtype, 0, 100, random_method='randint')
        return make_tensor(shape, dtype)

    def test_add_sub_mixed_dtype(
            self,
            input_dtypes,
            other_dtypes,
            *,
            grad_cmp=True,
    ):
        """Run add/sub tests with mixed tensor dtypes.

        Args:
            input_dtypes: Iterable of dtypes for the first input.
            other_dtypes: Iterable of dtypes for the second input.
            grad_cmp: Whether to run gradient comparison.
        """
        def add_sub_mixed_dtype_sample_inputs_func():
            op_params = [
                ((3,), (3,), -5.5),
                ((3, 0), (1, 0), 6),
                ((2, 3, 4), (2, 3, 1), 1.33),
                ((2, 3, 4, 2), (2,), 0.0)
            ]

            sample_inputs = []
            for input_dtype in input_dtypes:
                for other_dtype in other_dtypes:
                    for input_shape, other_shape, alpha in op_params:
                        input_tensor = self.get_tensor_by_dtype(input_shape, input_dtype)
                        other_tensor = self.get_tensor_by_dtype(other_shape, other_dtype)
                        if input_dtype in self.extra_uint_dtypes or other_dtype in self.extra_uint_dtypes:
                            alpha = abs(alpha)
                        if input_dtype in self.integer_dtypes or other_dtype in self.integer_dtypes:
                            alpha = int(alpha)
                        sample_inputs.append(OpSampleInput(
                            op_input=input_tensor,
                            op_args=(other_tensor,),
                            op_kwargs={"alpha": alpha},
                            op_name=f'mint_add_{input_dtype}_{other_dtype}_mixed_dtype',
                        ))
            return sample_inputs

        self.update_sample_inputs(add_sub_mixed_dtype_sample_inputs_func)
        self.forward_cmp()
        if grad_cmp:
            self.grad_cmp()

    def test_add_sub_scalar_tensor_mixed(
            self,
            scalar,
            dtypes,
            *,
            scalar_is_input=True,
            grad_cmp=True,
    ):
        """Run add/sub tests mixing scalar and tensor inputs.

        Args:
            scalar: The scalar value to combine with tensors.
            dtypes: Iterable of tensor dtypes to test.
            scalar_is_input: If True, scalar is first arg, else second.
            grad_cmp: Whether to run gradient comparison.
        """
        def add_sub_scalar_tensor_mixed_sample_inputs_func():
            op_params = [
                ((3,), -5.5),
                ((3, 0), 6),
                ((2, 3, 4), 0.0),
            ]

            sample_inputs = []
            for dtype in dtypes:
                for input_shape, alpha in op_params:
                    input_tensor = self.get_tensor_by_dtype(input_shape, dtype)
                    if input_tensor.dtype in self.extra_uint_dtypes:
                        alpha = abs(alpha)
                    if dtype in self.integer_dtypes:
                        alpha = int(alpha)
                    else:
                        alpha = float(alpha)
                    sample_inputs.append(OpSampleInput(
                        op_input=scalar if scalar_is_input else input_tensor,
                        op_args=(input_tensor if scalar_is_input else scalar,),
                        op_kwargs={"alpha": alpha} if alpha is not None else {},
                        op_name=f'mint_add_{dtype}_scalar_{"input" if scalar_is_input else "other"}',
                    ))
            return sample_inputs

        self.update_sample_inputs(add_sub_scalar_tensor_mixed_sample_inputs_func)
        self.forward_cmp()
        if grad_cmp:
            self.grad_cmp()
