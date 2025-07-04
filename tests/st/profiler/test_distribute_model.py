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
from mindspore import context
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_hccl_allreduce():
    """
    Feature: mpi run 2P case
    Description: mpi run 2P case
    Expectation: success
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    return_code = os.system("mpirun --allow-run-as-root -n 2 pytest -s hccl_profiler.py")
    assert return_code == 0
