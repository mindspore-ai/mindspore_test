# Copyright 2024 Huawei Technologies Co., Ltd
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
# import pytest
import mindspore as ms
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor, ops
from parallel.utils.utils import compile_net


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")

class Net(nn.Cell):
    """
    Examples:

    1. flip bit on each step:

    mask = ops.auto_generate.generate_eod_mask_v2(input, Tensor(0),  # the elements position of the tensor
                                                  bit_pos=0, # which bit of the element
                                                  steps=[1], # which step of the training, only list supported
                                                  error_mode='cycle' # specific or cycle)

    2.flip bit on specific steps, for example, 5,7,8:
    mask = ops.auto_generate.generate_eod_mask_v2(input, Tensor(0),
                                                  bit_pos=0,
                                                  steps=[4, 5, 8],
                                                  error_mode='specific')
    """
    def __init__(self):
        super(Net, self).__init__()
        self.cur_step = ms.Parameter(Tensor(-1., ms.int64))
        self.d_step = Tensor(1., ms.int64)
        self.generate_eod_mask = ops.auto_generate.GenerateEodMaskV2().shard(((2,), (), (), (), ()))

    def construct(self, input_tensor, ele_pos, seed, offset, start=0, steps=0, error_mode='cycle',
                  flip_mode='default', multiply_factor=0., bit_pos=0, flip_probability=0.):
        self.cur_step = self.cur_step + self.d_step
        return self.generate_eod_mask(input_tensor, ele_pos, self.cur_step, seed, offset,
                                      start, steps, error_mode, flip_mode, multiply_factor,
                                      bit_pos, flip_probability)


def run_generate_eod_mask_v2_on_step(data, ele_pos, start, steps, error_mode='specific',
                                     flip_mode='default', multiply_factor=0., flip_probability=0.,
                                     bit_pos=0, changed_poses=(0,)):
    """
    Feature: Test GenerateEodMaskV2.
    Description: Test multi dtype inputs
    Expectation: Successful graph compilation.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=2, global_rank=0)
    print(f"\nStart Testing, error_mode '{error_mode}', flip_mode '{flip_mode}', "
          f"flip_probability '{flip_probability}', ele_pos '{ele_pos}'")
    seed = Tensor(0, ms.int64)
    offset = Tensor(0, ms.int64)
    net = Net()
    compile_net(net, data, Tensor(ele_pos, ms.int64), seed, offset,
                start, steps, error_mode, flip_mode, multiply_factor,
                bit_pos, flip_probability)


def test_generate_eod_mask_v2():
    """
    Feature: test op GenerateEodMoaskV2
    Description: test op GenerateEodMoaskV2
    Expectation: expect results
    """
    context.set_context(mode=context.GRAPH_MODE,
                        device_target="Ascend",
                        jit_config={"jit_level": "O2"})
    test_data = Tensor([0.1, -0.2], dtype=ms.float32)
    run_generate_eod_mask_v2_on_step(test_data, ele_pos=0, start=0, steps=[1], error_mode='cycle',
                                     flip_mode="bitflip_designed", bit_pos=0, changed_poses=[0])
