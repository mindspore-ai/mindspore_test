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
from tests.mark_utils import arg_mark
import os

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_gpto_exec_order():
    """
    Feature: this test call gpto_net.py
    Description: this test use msrun to run the gpto test
    Expectation: the test should pass without any error
    """
    return_code = os.system(
        "msrun --worker_num=1 --local_worker_num=1 --master_addr=127.0.0.1 "
        "--master_port=10969 --join=True gpto_net.py"
    )

    assert return_code == 0


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dfs_exec_order():
    """
    Feature: this test call gpto_net.py
    Description: this test use msrun to run the gpto test
    Expectation: the test should pass without any error
    """
    return_code = os.system(
        "msrun --worker_num=1 --local_worker_num=1 --master_addr=127.0.0.1 "
        "--master_port=10969 --join=True dfs_net.py"
    )

    assert return_code == 0
