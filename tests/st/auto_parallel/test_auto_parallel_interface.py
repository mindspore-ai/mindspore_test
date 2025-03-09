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


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_msrun_pipeline_remove_redundancy():
    '''
    Feature: test remove redundancy with auto_parallel interface.
    Description: test pipeline net train and predict using msrun.
    Expectation: run success.
    '''
    return_code = os.system(
        "msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 "
        "--master_port=10801 --join=True --log_dir=./test_cpkt_pp2/auto_parallel/remove_redundancy_log "
        "pytest -s cpkt_rm_redundancy_auto_parallel.py"
    )
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level3", card_mark="allcards", essential_mark="essential")
def test_msrun_cpkt_transfer_functional():
    '''
    Feature: test checkpoints file transfer with auto_parallel interface.
    Description: test checkpoints file and strategy transfer with model or functional programming using msrun.
    Expectation: run success.
    '''
    return_code = os.system(
        "msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 "
        "--master_port=10801 --join=True --log_dir=./test_cpkt_transfer/cpkt_transfer_log "
        "pytest -s cpkt_transfer_functional_model_auto_parallel.py"
    )
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_msrun_transformer_opt():
    '''
    Feature: test transformer_opt in auto_parallel interface.
    Description: test ascend config set by transformer_opt using msrun.
    Expectation: run success.
    '''
    return_code = os.system(
        "msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 "
        "--master_port=10801 --join=True --log_dir=./test_ascend_config/ascend_config_log "
        "pytest -s transformer_opt_auto_parallel.py"
    )
    assert return_code == 0
