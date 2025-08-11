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

import numpy as np
import mindspore as ms
import mindspore.communication.management as D
from mindspore.parallel import Layout, custom_shard



def setup_module():
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="Ascend")


layout_dp = Layout(device_matrix=(2,), alias_name=("dp",))("dp")
layout_mp = Layout(device_matrix=(2,), alias_name=("mp",))("mp")
layout_dp_mp = Layout(device_matrix=(2, 2), alias_name=("dp", "mp"))("dp", "mp")


def create_dist_tensor(layout):
    """create_dist_tensor"""
    tensor = ms.Tensor(np.array([1.0, 2.0]), dtype=ms.float32)
    tensor = tensor.local_to_global(layout)
    return tensor

D.init()

def test_basic_functionality():
    '''
    Feature: Basic layout matching in custom_shard.
    Description: Test input/output layout assignment functionality.
    Expectation: Output tensors should match specified layouts without redistribution.
    '''
    def local_func(x, y):
        return x + y, x * y

    wrapped = custom_shard(
        func=local_func,
        out_layouts=(layout_dp, layout_mp),
        in_layouts=(layout_dp, layout_mp),
        redistribute_inputs=False
    )

    x = create_dist_tensor(layout_dp)
    y = create_dist_tensor(layout_mp)

    out1, out2 = wrapped(x, y)
    assert out1.layout == layout_dp
    assert out2.layout == layout_mp

def test_redistribution_enabled():
    '''
    Feature: Input redistribution in custom_shard.
    Description: Test automatic input redistribution when enabled.
    Expectation: Should accept differently-layouted input and produce correctly-layouted output.
    '''
    def local_func(x):
        return x * 2

    wrapped = custom_shard(
        func=local_func,
        out_layouts=(layout_mp,),
        in_layouts=(layout_dp,),
        redistribute_inputs=True
    )

    x = create_dist_tensor(layout_dp)
    out = wrapped(x)
    assert out.layout == layout_mp

def test_mixed_input_types():
    '''
    Feature: Mixed input types handling.
    Description: Test combination of distributed tensors and primitive types.
    Expectation: Should process mixed inputs and maintain layout for tensor outputs.
    '''
    def local_func(x, y, z):
        return x + z, y * z

    wrapped = custom_shard(
        func=local_func,
        out_layouts=(layout_dp, None),
        in_layouts=(layout_dp, None, None),
        redistribute_inputs=True
    )

    x = create_dist_tensor(layout_dp)
    y = 3.0
    z = 4.0

    out1, out2 = wrapped(x, y, z)
    assert isinstance(out1, ms.Tensor)
    assert out1.layout == layout_dp
    assert isinstance(out2, float)

def test_no_distributed_inputs():
    '''
    Feature: Non-distributed input handling.
    Description: Test behavior with only primitive-type inputs.
    Expectation: Should execute normally without layout operations.
    '''
    def local_func(a, b):
        return a + b, a * b

    wrapped = custom_shard(
        func=local_func,
        out_layouts=(layout_dp, layout_mp),
        in_layouts=(None, None),
        redistribute_inputs=False
    )

    out1, out2 = wrapped(3, 4)
    assert out1 == 7
    assert out2 == 12

def test_single_output_conversion():
    '''
    Feature: Single output conversion.
    Description: Test non-tuple single output handling with layout conversion.
    Expectation: Should correctly handle and convert single tensor output.
    '''
    def local_func(x):
        return x * 2

    wrapped = custom_shard(
        func=local_func,
        out_layouts=(layout_mp,),
        in_layouts=(layout_dp,),
        redistribute_inputs=True
    )

    x = create_dist_tensor(layout_dp)
    out = wrapped(x)
    assert isinstance(out, ms.Tensor)
    assert out.layout == layout_mp
