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
from mindspore import context
from mindspore.mint.distributed.distributed import (
    init_process_group,
    is_available,
    is_initialized,
    barrier,
)
#msrun --worker_num=8 --local_worker_num=8 --master_port=10923 --bind_core True --join True pytest -sv --disable-warnings  test_comm_init.py

context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
def test_is_available():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    ret = is_available()
    assert ret is True

def test_is_initialized():
    """
    Feature: test distributed op
    Description: test comm op in python native
    Expectation: success
    """
    ret = is_initialized()
    assert ret is False
    init_process_group()
    ret = is_initialized()
    assert ret is True
    barrier()
    ret = is_initialized()
    assert ret is True
