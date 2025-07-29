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
import os
from mindspore import jit, nn
from mindspore.communication import init

os.environ["MS_DEV_SAVE_GRAPHS"] = "1"
os.environ["MS_DEV_SAVE_GRAPHS_PATH"] = "./ir"
os.environ["MS_DEV_DUMP_IR_PASSES"] = "validate"


class Net(nn.Cell):
    @jit
    def construct(self, x):
        return x + 2

def test_ir_path_with_distributed_initialized():
    '''
    Feature: test ir save path.
    Description: Test ir save path of different rank using msrun.
    Expectation: Generate the expected ir path.
    '''
    init()

    x = Net()
    x(3)
