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

"""test func shard in python"""
import os
import mindspore as ms
from tests.mark_utils import arg_mark


def setup_module():
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="Ascend")


def run_case(case_name, master_port):
    cmd = f"export GLOG_v=2 && msrun --worker_num=8 --local_worker_num=8 " \
          f"--master_addr=127.0.0.1 --master_port={master_port} " \
          f"--join=True --log_dir=./{case_name} pytest -s -v " \
          f"func_shard_in_python.py::{case_name}"
    ret = os.system(cmd)
    assert ret == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="unessential")
def test_func_shard_1():
    '''
    Feature: run shard in python.
    Description: Test func shard in python.
    Expectation: Run success.
    '''
    case_name = "test_func_shard_1"
    master_port = 11292
    run_case(case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="unessential")
def test_func_shard_2():
    '''
    Feature: run shard in python.
    Description: Test func shard in python.
    Expectation: Run success.
    '''
    case_name = "test_func_shard_2"
    master_port = 11293
    run_case(case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="unessential")
def test_func_shard_3():
    '''
    Feature: run shard in python.
    Description: Test func shard in python.
    Expectation: Run success.
    '''
    case_name = "test_func_shard_3"
    master_port = 11294
    run_case(case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="unessential")
def test_func_shard_with_bprop():
    '''
    Feature: run shard in python.
    Description: Test func shard in python.
    Expectation: Run success.
    '''
    case_name = "test_func_shard_with_bprop"
    master_port = 11295
    run_case(case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="unessential")
def test_func_shard_linear_model_parallel():
    '''
    Feature: run shard in python.
    Description: Test linear func shard in python.
    Expectation: Run success.
    '''
    case_name = "test_linear_model_parallel"
    master_port = 11295
    run_case(case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="unessential")
def test_func_shard_linear_sequence_parallel():
    '''
    Feature: run shard in python.
    Description: Test linear func shard in python.
    Expectation: Run success.
    '''
    case_name = "test_linear_sequence_parallel"
    master_port = 11296
    run_case(case_name, master_port)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="unessential")
def test_func_shard_with_parameter_plan():
    '''
    Feature: run shard in python.
    Description: Test linear func shard in python.
    Expectation: Run success.
    '''
    case_name = "test_func_shard_with_parameter_plan"
    master_port = 11297
    run_case(case_name, master_port)
