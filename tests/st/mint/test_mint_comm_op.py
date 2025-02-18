
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
@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_hccl_mint_ops():
    """
    Feature: mpi run 8P case
    Description: mpi run 8P case
    Expectation: success
    """
    return_code = os.system(
        "msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 --master_port=10666 --join=True "\
        "pytest -s test_distributed.py"
    )
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_hccl_mint_object_ops():
    """
    Feature: mpi run 8P case
    Description: mpi run 8P case
    Expectation: success
    """
    return_code = os.system(
        "msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 --master_port=10666 --join=True "\
        "pytest -s test_comm_object.py"
    )
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_hccl_mint_cpu_ops():
    """
    Feature: mpi run 8P case
    Description: mpi run 8P case
    Expectation: success
    """
    return_code = os.system(
        "msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 --master_port=10666 --join=True "\
        "pytest -s test_comm_cpu.py"
    )
    assert return_code == 0

@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_hccl_mint_init_ops():
    """
    Feature: mpi run 8P case
    Description: mpi run 8P case
    Expectation: success
    """
    return_code = os.system(
        "msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 --master_port=10666 --join=True "\
        "pytest -s test_comm_init.py"
    )
    assert return_code == 0
