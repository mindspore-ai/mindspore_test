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
- UnaryOpsFactory: utilities to compare MindSpore unary ops with Benchmark
  references.
- Static and dynamic-shape parity checks, with optional gradient comparisons.
"""
import mindspore as ms
from tests.st.ops.share._internal.meta import OpsFactory
from tests.st.ops.share._op_info.op_info import OpInfo
from tqdm import tqdm


class UnaryOpsFactory(OpsFactory):
    """Factory for testing unary operations.

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
        # Ensure pylint knows _douts is defined in this class.
        self._douts = None

    def test_unary_op_reference(
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
            print(f"\nop_name: {self.op_name}, mode:{self._context_mode}, test_unary_op_reference...")
            if grad_cmp:
                self.supported_dtypes = tuple(d for d in self.supported_dtypes if d.is_floating_point)
            for dtype in tqdm(self.supported_dtypes):
                if grad_cmp:
                    for sample_input in self.op_basic_reference_inputs_func(self.op_info, dtype, device=self._device):
                        self.compare_with_torch(sample_inputs=sample_input, grad_cmp=True)
                else:
                    for sample_input in self.op_basic_reference_inputs_func(self.op_info, dtype, device=self._device):
                        self.compare_with_torch(sample_inputs=sample_input)
                    if self.op_extra_reference_inputs_func is not None:
                        for sample_input in self.op_extra_reference_inputs_func(
                                self.op_info,
                                dtype,
                                device=self._device,
                        ):
                            self.compare_with_torch(sample_inputs=sample_input)
        except Exception as e:
            print(f"\ntest_unary_op_reference failed:"
                  f"\nop_name: {self.op_name}"
                  f"\nmode: {self._context_mode}"
                  f"\ndtype: {dtype}"
                  f"\n{sample_input.summary(True)}")
            raise e

    def test_unary_op_dynamic(
        self,
        *,
        grad_cmp=False,
        only_dynamic_shape=False,
        only_dynamic_rank=False,
    ):
        """Run dynamic-shape tests against Benchmark.

        Args:
            grad_cmp: When True, also compare first-order gradients.
            only_dynamic_shape: If True, only run dynamic-shape cases (fixed rank).
            only_dynamic_rank: If True, only run dynamic-rank cases (shape varies in rank).
        """
        if self.op_info.op_dynamic_inputs_func is None:
            print(f"\nop_name: {self.op_name} has no op_dynamic_inputs_func, "
                  f"skip test_unary_op_dynamic.")
            return

        try:
            print(f"\nop_name: {self.op_name}, mode:{self._context_mode}, test_unary_op_dynamic...")
            for op_dynamic_input in self.op_info.op_dynamic_inputs_func(
                    self.op_info,
                    dtype=ms.float32,
                    device=self._device,
                    only_dynamic_shape=only_dynamic_shape,
                    only_dynamic_rank=only_dynamic_rank,
            ):
                if grad_cmp:
                    self.compare_with_torch_dynamic(op_dynamic_inputs=op_dynamic_input, grad_cmp=True)
                else:
                    self.compare_with_torch_dynamic(op_dynamic_inputs=op_dynamic_input)
        except Exception as e:
            print(f"\ntest_unary_op_dynamic failed:"
                  f"\nop_name: {self.op_name}"
                  f"\nmode: {self._context_mode}"
                  f"\n{op_dynamic_input.summary()}")
            raise e
