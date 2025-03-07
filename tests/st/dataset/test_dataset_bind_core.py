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
# ==============================================================================
"""
Testing dataset bind core
"""
import os
import pytest

from tests.mark_utils import arg_mark


@pytest.mark.parametrize("cleanup_temporary_files", ["./dataset_bind_out_1.log"], indirect=True)
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
def test_dataset_bind_core(cleanup_temporary_files):
    """
    Feature: Dataset bind core
    Description: Verify Binding functions
    Expectation: Output is equal to the expected output
    """
    os.environ['GLOG_v'] = str(1)
    real_path = os.path.realpath(os.getcwd())
    script = real_path + "/dataset_bind_core_base.py"
    output = real_path + "/dataset_bind_out_1.log"
    assert os.path.exists(script)

    cmd = (f"python {script} first_function > {output} 2>&1")
    os.system(cmd)

    assert os.path.exists(output)

    bind_str = "Current thread has been bound to core list"

    with open(output, "r") as f:
        output_log = f.read()

    assert bind_str in output_log


@pytest.mark.parametrize("cleanup_temporary_files", ["./dataset_bind_out_2.log"], indirect=True)
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
def test_dataset_bind_core_configuration(cleanup_temporary_files):
    """
    Feature: Dataset bind core
    Description: Verify that set_cpu_affinity and set_numa_enable are set at the same time
    Expectation: Output is equal to the expected output
    """
    os.environ['GLOG_v'] = str(1)
    real_path = os.path.realpath(os.getcwd())
    script = real_path + "/dataset_bind_core_base.py"
    output = real_path + "/dataset_bind_out_2.log"
    assert os.path.exists(script)

    cmd = (f"python {script} second_function > {output} 2>&1")
    os.system(cmd)

    assert os.path.exists(output)

    bind_str = "Start core binding"

    with open(output, "r") as f:
        output_log = f.read()

    assert bind_str in output_log


if __name__ == "__main__":
    test_dataset_bind_core(cleanup_temporary_files)
    test_dataset_bind_core_configuration(cleanup_temporary_files)
