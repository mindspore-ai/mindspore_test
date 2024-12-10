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
import subprocess
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
def test_bind_core_auto():
    """
    Feature: Runtime set_cpu_affinity api.
    Description: Test runtime.set_cpu_affinity api which automatically bind thread core.
    Expectation: Core bound for module and threads.
    """
    os.environ['GLOG_v'] = str(1)
    os.system("python test_bind_core_auto.py > ./auto.log 2>&1")
    result = subprocess.getoutput("grep 'bind core' ./auto.log")
    assert result.find("Module bind core policy generated:") != -1


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
def test_bind_core_manual():
    """
    Feature: Runtime set_cpu_affinity api.
    Description: Test runtime.set_cpu_affinity api which manually bind thread core.
    Expectation: Core bound for module and threads as input affinity_cpu_list.
    """
    os.environ['GLOG_v'] = str(1)
    os.system("python test_bind_core_manual.py > ./manual.log 2>&1")
    result = subprocess.getoutput("grep 'bind core' ./manual.log")
    manual_policy_str = ("Module bind core policy generated: {0: {'main': [0], 'runtime': [1, 2, 3, 4, 5], "
                         "'pynative': [1, 2, 3, 4], 'minddata': [6, 7, 8, 9]}, 1: {'main': [10], "
                         "'runtime': [11, 12, 13, 14, 15], 'pynative': [11, 12, 13, 14], 'minddata': "
                         "[16, 17, 18, 19]}, 2: {'main': [20], 'runtime': [21, 22, 23, 24, 25], 'pynative': "
                         "[21, 22, 23, 24], 'minddata': [26, 27, 28, 29]}, 3: {'main': [30], 'runtime': "
                         "[31, 32, 33, 34, 35], 'pynative': [31, 32, 33, 34], 'minddata': [36, 37, 38, 39]}, "
                         "4: {'main': [40], 'runtime': [41, 42, 43, 44, 45], 'pynative': [41, 42, 43, 44], "
                         "'minddata': [46, 47, 48, 49]}, 5: {'main': [50], 'runtime': [51, 52, 53, 54, 55], "
                         "'pynative': [51, 52, 53, 54], 'minddata': [56, 57, 58, 59]}, 6: {'main': [60], 'runtime': "
                         "[61, 62, 63, 64, 65], 'pynative': [61, 62, 63, 64], 'minddata': [66, 67, 68, 69]}, "
                         "7: {'main': [70], 'runtime': [71, 72, 73, 74, 75], 'pynative': [71, 72, 73, 74], "
                         "'minddata': [76, 77, 78, 79]}}")
    assert result.find(manual_policy_str) != -1
