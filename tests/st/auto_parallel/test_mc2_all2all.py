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
def test_mc2_alltoall_allgather_batchmatmul_withoutsilu():
    '''
    Feature: MC2 fusion.
    Description: Test alltoall-allgather-batchmatmul without silu fusion.
    Expectation: Run success
    '''
    ret = os.system(
        f"msrun --worker_num=8 --local_worker_num=8 --log_dir=msrun_log --join=True --master_port=8129\
            pytest -s mc2_all2all.py::test_mc2_alltoall_allgather_batchmatmul_withoutsilu"
    )
    assert ret == 0


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_mc2_alltoall_allgather_batchmatmul_withsilu():
    '''
    Feature: MC2 fusion.
    Description: Test Test alltoall-allgather-batchmatmul with silu fusion.
    Expectation: Run success
    '''
    ret = os.system(
        f"msrun --worker_num=8 --local_worker_num=8 --log_dir=msrun_log --join=True --master_port=8139\
            pytest -s mc2_all2all.py::test_mc2_alltoall_allgather_batchmatmul_withsilu"
    )
    assert ret == 0
