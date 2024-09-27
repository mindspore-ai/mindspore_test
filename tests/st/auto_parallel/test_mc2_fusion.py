# Copyright 2022 Huawei Technologies Co., Ltd
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


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_all_gather_matmul_forward():
    '''
    Feature: MC2 fusion.
    Description: Test all_gather-matmul fusion in forward.
    Expectation: Run success
    '''
    ret = os.system("mpirun -n 8 pytest -s mc2_fusion.py::test_all_gather_matmul_forward")
    assert ret == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_matmul_reduce_scatter_forward():
    '''
    Feature: MC2 fusion.
    Description: Test matmul-reduce_scatter fusion in forward.
    Expectation: Run success
    '''
    ret = os.system("mpirun -n 8 -x HCCL_DETERMINISTIC=true "
                    "pytest -s mc2_fusion.py::test_matmul_reduce_scatter_forward")
    assert ret == 0

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_all_gather_matmul_enable_all_kbk_mode():
    '''
    Feature: MC2 fusion.
    Description: Test all_gather_matmul_enable_all_kbk_mode.
    Expectation: Run success
    '''
    ret = os.system("mpirun -n 8 -x HCCL_DETERMINISTIC=true "
                    "pytest -s mc2_fusion.py::test_all_gather_matmul_enable_all_kbk_mode")
    assert ret == 0

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_matmul_reduce_scatter_enable_all_kbk_mode():
    '''
    Feature: MC2 fusion.
    Description: Test matmul_reduce_scatter_enable_all_kbk_mode.
    Expectation: Run success
    '''
    ret = os.system("mpirun -n 8 -x HCCL_DETERMINISTIC=true "
                    "pytest -s mc2_fusion.py::test_matmul_reduce_scatter_enable_all_kbk_mode")
    assert ret == 0

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_matmul_all_reduce_enable_all_kbk_mode():
    '''
    Feature: MC2 fusion.
    Description: Test matmul_all_reduce_enable_all_kbk_mode.
    Expectation: Run success
    '''
    ret = os.system("mpirun -n 8 -x HCCL_DETERMINISTIC=true "
                    "pytest -s mc2_fusion.py::test_matmul_all_reduce_enable_all_kbk_mode")
    assert ret == 0
