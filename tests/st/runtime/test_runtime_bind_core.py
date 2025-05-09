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
import pytest
import subprocess
import mindspore as ms
from tests.mark_utils import arg_mark


def _check_env_valid_cpu_resource():
    try:
        result = subprocess.run(
            ["cat", "/sys/fs/cgroup/cpuset/cpuset.cpus"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"Available CPU range is {result}.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Fail to execute command 'cat /sys/fs/cgroup/cpuset/cpuset.cpus', because {e}")
        return False


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
def test_bind_core_auto():
    """
    Feature: Runtime set_cpu_affinity api.
    Description: Test runtime.set_cpu_affinity api which automatically bind thread core.
    Expectation: Core bound for module and threads.
    """
    if _check_env_valid_cpu_resource():
        os.environ['GLOG_v'] = str(1)
        real_path = os.path.realpath(os.getcwd())
        script = real_path + "/test_bind_core_auto.py"
        output = real_path + "/auto.log"
        assert os.path.exists(script)

        cmd = (f"python {script} > {output} 2>&1")
        os.system(cmd)

        assert os.path.exists(output)
        with open(output, "r") as f:
            output_log = f.read()
            print(output_log, flush=True)
        assert "Module bind core policy generated:" in output_log
    else:
        print("Skip this ST, as the environment is not suitable for thread bind core.")


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
def test_bind_core_manual():
    """
    Feature: Runtime set_cpu_affinity api.
    Description: Test runtime.set_cpu_affinity api which manually bind thread core.
    Expectation: Core bound for module and threads as input affinity_cpu_list and module_to_cpu_dict.
    """
    if _check_env_valid_cpu_resource():
        os.environ['GLOG_v'] = str(1)
        real_path = os.path.realpath(os.getcwd())
        script = real_path + "/test_bind_core_manual.py"
        output = real_path + "/manual.log"
        assert os.path.exists(script)

        cmd = (f"python {script} > {output} 2>&1")
        os.system(cmd)

        assert os.path.exists(output)
        with open(output, "r") as f:
            output_log = f.read()
            print(output_log, flush=True)
        manual_policy_str = ("Module bind core policy generated: {'main': [0, 1, 2, 3], "
                             "'minddata': [4, 5, 6, 7], 'pynative': [10, 21]}")
        assert manual_policy_str in output_log
    else:
        print("Skip this ST, as the environment is not suitable for thread bind core.")


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
def test_bind_core_manual_no_available_cpus():
    """
    Feature: Runtime set_cpu_affinity api.
    Description: Test runtime.set_cpu_affinity api manually input unavailable cpu list.
    Expectation: RuntimeError reported.
    """
    affinity_cpu_list = ["300-500"]
    with pytest.raises(RuntimeError) as err_info:
        ms.runtime.set_cpu_affinity(True, affinity_cpu_list)
    assert f"set in affinity_cpu_list:{affinity_cpu_list} is not available." in str(err_info)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
def test_bind_core_repeatly_call():
    """
Feature: Runtime set_cpu_affinity api.
    Description: Test runtime.set_cpu_affinity api repeatedly called..
    Expectation: RuntimeError reported.
    """
    affinity_cpu_list = ["0-10"]
    ms.runtime.set_cpu_affinity(True, affinity_cpu_list)
    with pytest.raises(RuntimeError) as err_info:
        ms.runtime.set_cpu_affinity(False)
    assert "The 'mindspore.runtime.set_cpu_affinity' cannot be set repeatedly." in str(err_info)
