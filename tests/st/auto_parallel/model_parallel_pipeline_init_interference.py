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


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level3", card_mark="allcards", essential_mark="unessential")
def test_msrun_pipeline_remove_redundancy_auto_parallel():
    '''
    Feature: test custom op parallel
    Description: Test a net that consists of 10 sharded matmul ops using msrun.
    Expectation: Run success; results before and after enabling this feature should be the same.
    '''
    return_code = os.system(
        "msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 "
        "--master_port=10801 --join=True --log_dir=./train_pp/auto_parallel/test_pipeline_log "
        "pytest -s pipeline_cpkt_auto_parallel_interface.py"
    )
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level3", card_mark="allcards", essential_mark="unessential")
def test_msrun_pipeline_remove_redundancy_context():
    '''
    Feature: test custom op parallel
    Description: Test a net that consists of 10 sharded matmul ops using msrun.
    Expectation: Run success; results before and after enabling this feature should be the same.
    '''
    return_code = os.system(
        "msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 "
        "--master_port=10801 --join=True --log_dir=./train_pp/context/test_pipeline_log "
        "pytest -s pipeline_cpkt_context_interface.py"
    )
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level3", card_mark="allcards", essential_mark="unessential")
def test_msrun_model_parallel_mp_2_1_1_2_pp():
    '''
    Feature: Model parallel, strategy:((2,1),(1,2)).
    Description: Test model parallel.
    Expectation: Run success.
    '''
    ret = os.system("export GLOG_v=2 && msrun --worker_num=8 --local_worker_num=8 "
                    "--master_addr=127.0.0.1 --master_port=10807 "
                    "--join=True --log_dir=./model_parallel_logs/strategy_2_1_1_2 pytest -s -v "
                    "model_parallel.py::test_parallel_mp_compare_context_autoparallel_pipeline_config")
    assert ret == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level3", card_mark="allcards", essential_mark="unessential")
def test_msrun_pipeline_remove_redundancy_auto_parallel_init():
    '''
    Feature: test custom op parallel
    Description: Test a net that consists of 10 sharded matmul ops using msrun.
    Expectation: Run success; results before and after enabling this feature should be the same.
    '''
    return_code = os.system(
        "msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 "
        "--master_port=10801 --join=True --log_dir=./train_pp/auto_parallel/test_pipeline_log "
        "pytest -s model_parallel_pipeline_init.py"
    )
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level3", card_mark="allcards", essential_mark="unessential")
def test_msrun_model_parallel_mp_2_1_1_2_pp_lazy_init():
    '''
    Feature: Model parallel, strategy:((2,1),(1,2)).
    Description: Test model parallel.
    Expectation: Run success.
    '''
    ret = os.system("export GLOG_v=2 && msrun --worker_num=8 --local_worker_num=8 "
                    "--master_addr=127.0.0.1 --master_port=10807 "
                    "--join=True --log_dir=./model_parallel_logs/strategy_2_1_1_2 pytest -s -v "
                    "model_parallel_pipeline.py::test_parallel_mp_compare_context_auto_pp_config_lazy_init")
    assert ret == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level3", card_mark="allcards", essential_mark="unessential")
def test_msrun_model_parallel_mp_2_1_1_2_pp_lazy_init_dp():
    '''
    Feature: Model parallel, strategy:((2,1),(1,2)).
    Description: Test model parallel.
    Expectation: Run success.
    '''
    ret = os.system("export GLOG_v=2 && msrun --worker_num=8 --local_worker_num=8 "
                    "--master_addr=127.0.0.1 --master_port=10807 "
                    "--join=True --log_dir=./model_parallel_logs/strategy_2_1_1_2 pytest -s -v "
                    "model_parallel_pipeline.py::test_parallel_mp_compare_context_auto_pp_config_lazy_init_dp")
    assert ret == 0

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level3", card_mark="allcards", essential_mark="unessential")
def test_msrun_model_parallel_mp_functional_programming():
    '''
    Feature: Model parallel, strategy:((1,1),(1,2)).
    Description: Test model parallel.
    Expectation: The error of the loss is within the allowable range.
    '''
    ret = os.system("export GLOG_v=2 && msrun --worker_num=8 --local_worker_num=8 "
                    "--master_addr=127.0.0.1 --master_port=10807 "
                    "--join=True --log_dir=./model_parallel_functional_programming_logs/strategy_1_1_1_2 pytest -s -v "
                    "model_parallel_pipeline_init.py::test_parallel_mp_compare_context_auto_fun_programming")
    assert ret == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level3", card_mark="allcards", essential_mark="unessential")
def test_msrun_model_parallel_mp_2_1_1_2_pp_lazy_init_data_sink():
    '''
    Feature: Model parallel, strategy:((2,1),(1,2)).
    Description: Test model parallel.
    Expectation: The error of the loss is within the allowable range.
    '''
    ret = os.system("export GLOG_v=2 && msrun --worker_num=8 --local_worker_num=8 "
                    "--master_addr=127.0.0.1 --master_port=10807 "
                    "--join=True --log_dir=./model_parallel_functional_programming_logs/strategy_1_1_1_2 pytest -s -v "
                    "model_parallel_pipeline.py::test_parallel_mp_compare_context_auto_pp_cfg_lazy_init_inline_sink")
    assert ret == 0
