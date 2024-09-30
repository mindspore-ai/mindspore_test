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

os.environ['HCCL_IF_BASE_PORT'] = '30000'

@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_graph_mode_parallel_complex_input():
    '''
    Feature: Parallel Support for Complex64 input
    Description: graph mode
    Expectation: Run success
    '''
    os.environ['ASCEND_GLOBAL_EVENT_ENABLE'] = str(1)
    os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = str(0)
    os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = str(1)
    os.environ['GLOG_v'] = str(1)
    # print("netstat")
    # os.system(f"netstat")
    print("netstat -tunlp")
    os.system("netstat -tunlp")
    os.system("netstat -tunlp > netstat.txt")


    ret = os.system("mpirun -n 8 --allow-run-as-root pytest -s -v parallel_complex_input.py::test_graph_mode \
                    > parallel_complex_input_train.log 2>&1")
    if ret != 0:
        import datetime
        t = datetime.datetime.now()
        f = t.strftime('%m-%d-%H:%M:%S')
        os.system(f"mkdir ~/parallel_complex_input_{f} && cp -rf *.log ~/parallel_complex_input_{f} \
                    && cp -rf ~/ascend/log ~/parallel_complex_input_{f} \
                    && cp -rf netstat.txt ~/parallel_complex_input_{f}")
    assert ret == 0
