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
        self.op1 = mint.sin
        self.op2 = mint.cos

    def construct(self, x):
        out = self.op1(x)
        out = self.op2(out)
        out = self.op1(out)
        out = self.op2(out)
        out = self.op1(out)
        out = self.op2(out)
        out = self.op1(out)
        return out

def test_kbyk_aclnn_cache_1():
    """
    Feature: test aclnn cache.
    Description: set global aclnn cache
    Expectation: set aclnn cache failed
    """
    x = mindspore.Tensor(np.ones([1, 3, 224, 224]).astype(np.float32))
    context.set_context(mode=mindspore.GRAPH_MODE, device_target="Ascend", jit_level="O0")
    mindspore.device_context.ascend.op_tuning.aclnn_cache(True)
    net = Net()
    for _ in range(5):
        net(x)


def test_kbyk_aclnn_cache_2():
    """
    Feature: test aclnn cache.
    Description: set global aclnn cache
    Expectation: set aclnn cache failed
    """
    x = mindspore.Tensor(np.ones([1, 3, 224, 224]).astype(np.float32))
    context.set_context(mode=mindspore.GRAPH_MODE, device_target="Ascend", jit_level="O0")
    mindspore.device_context.ascend.op_tuning.aclnn_cache(cache_queue_length=100)
    net = Net()
    for _ in range(5):
        net(x)


def test_pyboost_aclnn_cache():
    """
    Feature: test aclnn cache.
    Description: set global aclnn cache
    Expectation: set aclnn cache failed
    """
    x = mindspore.Tensor(np.ones([1, 3, 224, 224]).astype(np.float32))
    context.set_context(mode=mindspore.PYNATIVE_MODE, device_target="Ascend")
    net = Net()
    for _ in range(5):
        net(x)

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_aclnn_cache_length_kbyk():
    """
    Feature: aclnn cache
    Description: set aclnn cache length to 100
    Expectation: set aclnn cache length failed
    """
    os.environ["VLOG_v"] = "20002"
    os.system("pytest -sv test_aclnn_cache.py::test_kbyk_aclnn_cache_2 > log_cache_kbyk.txt 2>&1")
    ret = os.system("grep -i 'Set aclnn cache queue length of kbyk to 100' log_cache_kbyk.txt")
    assert ret == 0
    os.system("rm -rf log_cache_kbyk.txt")
    del os.environ["VLOG_v"]

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_global_aclnn_cache_kbyk():
    """
    Feature: global aclnn cache
    Description: set global aclnn cache
    Expectation: set global aclnn cache failed
    """
    # use ms global aclnn cache
    os.environ["VLOG_v"] = "20002"
    os.system("pytest -sv test_aclnn_cache.py::test_kbyk_aclnn_cache_1 > log_cache_kbyk.txt 2>&1")
    ret_miss = os.popen("grep -i ' gen executor miss cache, hash id: ' log_cache_kbyk.txt | wc -l").read()
    assert int(ret_miss.strip()) == 2
    os.system("rm -rf log_cache_kbyk.txt")
    del os.environ["VLOG_v"]

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_global_aclnn_cache_pyboost():
    """
    Feature: global aclnn cache
    Description: set global aclnn cache
    Expectation: set global aclnn cache failed
    """
    # use ms global aclnn cache
    os.environ["VLOG_v"] = "20002"
    os.system("pytest -sv test_aclnn_cache.py::test_pyboost_aclnn_cache > log_cache_pyboost.txt 2>&1")
    ret_miss = os.popen("grep -i 'miss cache, with hash id:' log_cache_pyboost.txt | wc -l").read()
    assert int(ret_miss.strip()) == 2
    os.system("rm -rf log_cache_pyboost.txt")
    del os.environ["VLOG_v"]
