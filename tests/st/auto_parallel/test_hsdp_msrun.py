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
from tests.mark_utils import arg_mark

@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_pure_dp():
    '''
    Feature: pure data parallel with hsdp api.
    Description: pure data parallel.
    Expectation: Run success
    '''
    ret = os.system(
        f"msrun --worker_num=8 --local_worker_num=8 --log_dir=msrun_log --join=True --master_port=18181\
            pytest -s test_hsdp.py::test_pure_dp"
    )
    assert ret == 0

@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_zero1_fully_shard():
    '''
    Feature: zero1 fully shard data parallel with hsdp api.
    Description: zero1 data parallel.
    Expectation: Run success
    '''
    ret = os.system(
        f"msrun --worker_num=8 --local_worker_num=8 --log_dir=msrun_log --join=True --master_port=18181\
            pytest -s test_hsdp.py::test_zero1_fully_shard"
    )
    assert ret == 0

@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_zero1_partial_shard():
    '''
    Feature: zero1 partial shard data parallel with hsdp api.
    Description: zero1 data parallel.
    Expectation: Run success
    '''
    ret = os.system(
        f"msrun --worker_num=8 --local_worker_num=8 --log_dir=msrun_log --join=True --master_port=18181\
            pytest -s test_hsdp.py::test_zero1_partial_shard"
    )
    assert ret == 0

@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_zero2_fully_shard():
    '''
    Feature: zero2 fully shard data parallel with hsdp api.
    Description: zero2 data parallel.
    Expectation: Run success
    '''
    ret = os.system(
        f"msrun --worker_num=8 --local_worker_num=8 --log_dir=msrun_log --join=True --master_port=18181\
            pytest -s test_hsdp.py::test_zero2_fully_shard"
    )
    assert ret == 0

@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_zero2_partial_shard():
    '''
    Feature: zero2 partial shard data parallel with hsdp api.
    Description: zero2 data parallel.
    Expectation: Run success
    '''
    ret = os.system(
        f"msrun --worker_num=8 --local_worker_num=8 --log_dir=msrun_log --join=True --master_port=18181\
            pytest -s test_hsdp.py::test_zero2_partial_shard"
    )
    assert ret == 0

@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_zero3_fully_shard():
    '''
    Feature: zero3 fully shard data parallel with hsdp api.
    Description: zero3 data parallel.
    Expectation: Run success
    '''
    ret = os.system(
        f"msrun --worker_num=8 --local_worker_num=8 --log_dir=msrun_log --join=True --master_port=18181\
            pytest -s test_hsdp.py::test_zero3_fully_shard"
    )
    assert ret == 0

@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_zero3_partial_shard():
    '''
    Feature: zero3 partial shard data parallel with hsdp api.
    Description: zero3 data parallel.
    Expectation: Run success
    '''
    ret = os.system(
        f"msrun --worker_num=8 --local_worker_num=8 --log_dir=msrun_log --join=True --master_port=18181\
            pytest -s test_hsdp.py::test_zero3_partial_shard"
    )
    assert ret == 0

@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_pure_dp_with_acc_grad():
    '''
    Feature: pure data parallel with grad accumulation.
    Description: pure data parallel.
    Expectation: Run success
    '''
    ret = os.system(
        f"msrun --worker_num=8 --local_worker_num=8 --log_dir=msrun_log --join=True --master_port=18181\
            pytest -s test_hsdp.py::test_pure_dp_with_acc_grad"
    )
    assert ret == 0

@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_zero1_fully_shard_with_acc_grad():
    '''
    Feature: zero1 fully shard data parallel with grad accumulation.
    Description: zero1 data parallel.
    Expectation: Run success
    '''
    ret = os.system(
        f"msrun --worker_num=8 --local_worker_num=8 --log_dir=msrun_log --join=True --master_port=18181\
            pytest -s test_hsdp.py::test_zero1_fully_shard_with_acc_grad"
    )
    assert ret == 0

@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_zero1_partial_shard_with_acc_grad():
    '''
    Feature: zero1 partial shard data parallel with grad accumulation.
    Description: zero1 data parallel.
    Expectation: Run success
    '''
    ret = os.system(
        f"msrun --worker_num=8 --local_worker_num=8 --log_dir=msrun_log --join=True --master_port=18181\
            pytest -s test_hsdp.py::test_zero1_partial_shard_with_acc_grad"
    )
    assert ret == 0

@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_zero2_fully_shard_with_acc_grad():
    '''
    Feature: zero2 fully shard data parallel with grad accumulation.
    Description: zero2 data parallel.
    Expectation: Run success
    '''
    ret = os.system(
        f"msrun --worker_num=8 --local_worker_num=8 --log_dir=msrun_log --join=True --master_port=18181\
            pytest -s test_hsdp.py::test_zero2_fully_shard_with_acc_grad"
    )
    assert ret == 0

@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_zero2_partial_shard_with_acc_grad():
    '''
    Feature: zero2 partial shard data parallel with grad accumulation.
    Description: zero2 data parallel.
    Expectation: Run success
    '''
    ret = os.system(
        f"msrun --worker_num=8 --local_worker_num=8 --log_dir=msrun_log --join=True --master_port=18181\
            pytest -s test_hsdp.py::test_zero2_partial_shard_with_acc_grad"
    )
    assert ret == 0

@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_zero3_fully_shard_with_acc_grad():
    '''
    Feature: zero3 fully shard data parallel with grad accumulation.
    Description: zero3 data parallel.
    Expectation: Run success
    '''
    ret = os.system(
        f"msrun --worker_num=8 --local_worker_num=8 --log_dir=msrun_log --join=True --master_port=18181\
            pytest -s test_hsdp.py::test_zero3_fully_shard_with_acc_grad"
    )
    assert ret == 0

@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_zero3_partial_shard_with_acc_grad():
    '''
    Feature: zero3 partial shard data parallel with grad accumulation.
    Description: zero3 data parallel.
    Expectation: Run success
    '''
    ret = os.system(
        f"msrun --worker_num=8 --local_worker_num=8 --log_dir=msrun_log --join=True --master_port=18181\
            pytest -s test_hsdp.py::test_zero3_partial_shard_with_acc_grad"
    )
    assert ret == 0
