#!/usr/bin/env python3
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
"""
Tests for other operations (non-unary/binary), e.g., chunk/gather.
"""
import pytest
from tests.mark_utils import arg_mark
from tests.st.ops.share._internal.meta import OpsFactory
from tests.st.ops.share._op_info.op_database import get_op_info, other_op_db


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("mode", ['pynative'])
@pytest.mark.parametrize("op_info", other_op_db)
def test_other_ops_reference_forward(mode, op_info):
    '''
    Feature: Other operations
    Description: Compare forward.
    Expectation: MindSpore matches Benchmark for outputs.
    '''
    fact = OpsFactory(
        op_info=get_op_info(op_info),
    )
    fact.set_context_mode(mode=mode)
    fact.test_op_reference()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("mode", ['pynative'])
@pytest.mark.parametrize("op_info", other_op_db)
def test_other_ops_reference_backward(mode, op_info):
    '''
    Feature: Other operations
    Description: Compare gradients.
    Expectation: MindSpore matches Benchmark for gradients.
    '''
    fact = OpsFactory(
        op_info=get_op_info(op_info),
    )
    fact.set_context_mode(mode=mode)
    fact.test_op_reference(grad_cmp=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("mode", ['kbk'])
@pytest.mark.parametrize("op_info", ['mint.chunk', 'mint.gather'])
def test_other_ops_dynamic_forward(mode, op_info):
    '''
    Feature: Other operations
    Description: Dynamic shape forward.
    Expectation: Outputs match Benchmark.
    '''
    fact = OpsFactory(
        op_info=get_op_info(op_info),
    )
    fact.set_context_mode(mode=mode)
    fact.test_op_dynamic(only_dynamic_shape=True)
    fact.test_op_dynamic(only_dynamic_rank=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b',
                      'platform_gpu',
                      'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("mode", ['kbk'])
@pytest.mark.parametrize("op_info", ['mint.chunk', 'mint.gather'])
def test_other_ops_dynamic_backward(mode, op_info):
    '''
    Feature: Other operations
    Description: Dynamic shape backward.
    Expectation: Gradients match Benchmark.
    '''
    fact = OpsFactory(
        op_info=get_op_info(op_info),
    )
    fact.set_context_mode(mode=mode)
    fact.test_op_dynamic(only_dynamic_shape=True, grad_cmp=True)
    fact.test_op_dynamic(only_dynamic_rank=True, grad_cmp=True)
