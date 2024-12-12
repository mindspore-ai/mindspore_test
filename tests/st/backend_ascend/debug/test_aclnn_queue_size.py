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
import os
from tests.mark_utils import arg_mark

import numpy as np
import mindspore
from mindspore import mint, context
from mindspore.nn import Cell


class Net(Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.op = mint.clone

    def construct(self, x):
        out = self.op(x)
        out = self.op(out)
        return out

def test_clone_pyboost():
    """
    Feature: test aclnn cache queue.
    Description: set aclnn cache queue failed
    Expectation: success set aclnn cache queue
    """
    x = mindspore.Tensor(np.ones([1, 3, 224, 224]).astype(np.float32))
    context.set_context(mode=mindspore.PYNATIVE_MODE, device_target="Ascend")
    net = Net()
    for _ in range(5):
        net(x)

def test_clone_kbyk():
    """
    Feature: test aclnn cache queue.
    Description: set aclnn cache queue failed
    Expectation: success set aclnn cache queue
    """
    x = mindspore.Tensor(np.ones([1, 3, 224, 224]).astype(np.float32))
    context.set_context(mode=mindspore.GRAPH_MODE, device_target="Ascend", jit_level="O0")
    net = Net()
    net(x)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_aclnn_queue_length_pyboost():
    """
    Feature: test aclnn cache queue.
    Description: set aclnn cache queue failed
    Expectation: success set aclnn cache queue
    """
    os.environ["MS_DEV_RUNTIME_CONF"] = "aclnn_cache_queue_length:0"
    os.environ["GLOG_v"] = "1"
    os.system("pytest -sv test_aclnn_queue_size.py::test_clone_pyboost > log_pyboost.txt 2>&1")
    ret = os.system("grep -i 'Set aclnn cache queue length of pyboost to 0' log_pyboost.txt")
    assert ret == 0
    os.system("rm -rf log_pyboost.txt")
    del os.environ["MS_DEV_RUNTIME_CONF"]
    del os.environ["GLOG_v"]


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_aclnn_queue_length_kbyk():
    """
    Feature: test aclnn cache queue.
    Description: set aclnn cache queue failed
    Expectation: success set aclnn cache queue
    """
    os.environ["MS_DEV_RUNTIME_CONF"] = "aclnn_cache_queue_length:0"
    os.environ["GLOG_v"] = "1"
    os.system("pytest -sv test_aclnn_queue_size.py::test_clone_kbyk > log_kbyk.txt 2>&1")
    ret = os.system("grep -i 'Set aclnn cache queue length of kbyk to 0' log_kbyk.txt")
    assert ret == 0
    os.system("rm -rf log_kbyk.txt")
    del os.environ["MS_DEV_RUNTIME_CONF"]
    del os.environ["GLOG_v"]
