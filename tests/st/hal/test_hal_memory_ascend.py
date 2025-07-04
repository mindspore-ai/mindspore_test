# Copyright 2024-2025 Huawei Technologies Co., Ltd
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
from mindspore import Tensor
import mindspore as ms
import mindspore.nn as nn
from mindspore.ops import operations as P
from tests.mark_utils import arg_mark
from tests.device_utils import set_device, get_device_id
import shutil
import os
import numpy as np
import acl

GB_TO_BYTE = 1024 << 20
FLOAT32_SIZE = 4


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.ops = P.Abs()

    def construct(self, x):
        return self.ops(x)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_empty_cache_vmm():
    """
    Feature: runtime memory api.
    Description: Test runtime memory empty cache api.
    Expectation: runtime.empty_cache api performs as expected.
    """
    set_device()
    os.environ['MS_ALLOC_CONF'] = "enable_vmm:true"

    net = Net()
    net(Tensor(2.0))
    reserved_size = ms.runtime.memory_reserved()
    assert reserved_size > 0
    ms.runtime.empty_cache()
    reserved_size = ms.runtime.memory_reserved()
    assert reserved_size == 0


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_empty_cache_without_vmm():
    """
    Feature: runtime memory api.
    Description: Test runtime memory empty cache api.
    Expectation: runtime.empty_cache api performs as expected.
    """
    set_device()
    os.environ['MS_ALLOC_CONF'] = "enable_vmm:false"
    for _ in range(1000):
        net = Net()
        net(Tensor(2.0))
        reserved_size = ms.runtime.memory_reserved()
        assert reserved_size > 0
        ms.runtime.empty_cache()
        reserved_size = ms.runtime.memory_reserved()
        assert reserved_size == 0


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_empty_cache_dryrun():
    """
    Feature: runtime memory api.
    Description: Test runtime memory empty cache api.
    Expectation: runtime.empty_cache api performs as expected.
    """
    set_device()
    os.environ["MS_SIMULATION_LEVEL"] = "1"
    os.environ["RANK_SIZE"] = "1"
    os.environ["RANK_ID"] = "0"

    net = Net()
    net(Tensor(2.0))
    reserved_size = ms.runtime.memory_reserved()
    assert reserved_size > 0
    ms.runtime.empty_cache()
    reserved_size = ms.runtime.memory_reserved()
    assert reserved_size == 0


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_memory_replay():
    """
    Feature: runtime memory api.
    Description: Test runtime memory replay api.
    Expectation: success.
    """
    mem_tracker_path = "test_replay_mem_tracker"
    try:
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        tracker_path = os.path.join(cur_dir, mem_tracker_path)
        os.environ['MS_ALLOC_CONF'] = "enable_vmm:false"
        cmd = "python hal_dryrun_case.py"
        ret = os.system(cmd)
        assert ret == 0
        ms.runtime.memory_replay(os.path.join(tracker_path, "memory_block.csv"))
    except Exception as e:
        remove_dir = ["kernel_meta", "offload", tracker_path]
        for d in remove_dir:
            if os.path.isdir(d):
                shutil.rmtree(d)
        raise e


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_huge_page_reserve_vmm():
    """
    Feature: reserve huge page for vmm.
    Description: Test whether huge page memory is reserved for vmm.
    Expectation: When huge page is reserved, use normal memory.
    """
    acl_hbm_mem_huge = 4
    set_device()
    os.environ['MS_ALLOC_CONF'] = "enable_vmm:false"
    device_id = get_device_id()
    acl.rt.set_device(int(device_id))

    huge_page_free, _, ret = acl.rt.get_mem_info(acl_hbm_mem_huge)
    assert ret == 0
    huge_page_free_gb = huge_page_free / GB_TO_BYTE
    huge_page_available_size_gb = 0.5
    huge_page_reserve_gb = max(huge_page_free_gb - huge_page_available_size_gb, 0)
    test_tensor_size_gb = huge_page_available_size_gb * 2
    tensor_element_num = int(GB_TO_BYTE * test_tensor_size_gb / FLOAT32_SIZE)
    ms.runtime.set_memory(huge_page_reserve_size=f"{huge_page_reserve_gb}GB")

    net = Net()
    result = net(Tensor(np.random.rand(tensor_element_num), ms.float32))
    result.asnumpy()
    huge_page_free, _, ret = acl.rt.get_mem_info(acl_hbm_mem_huge)
    assert ret == 0
    huge_page_free_gb = huge_page_free / GB_TO_BYTE
    assert huge_page_free_gb >= huge_page_reserve_gb
