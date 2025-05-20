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
import subprocess
from tests.mark_utils import arg_mark


def generate_dyn(file_name, func_name, dyn_file_name):
    if os.path.exists(dyn_file_name):
        os.remove(dyn_file_name)
    assert not os.path.exists(dyn_file_name)

    cmd = f"VLOG_v=1 python " + file_name + " " + func_name + " > " + dyn_file_name + " 2>&1"
    subprocess.check_output(cmd, shell=True)
    assert os.path.exists(dyn_file_name)
    with open(dyn_file_name, "r") as v_file:
        data = v_file.read()

    assert data.count("Start compiling") == 3
    assert data.count("End compiling") == 3
    os.remove(dyn_file_name)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_arguments_1():
    """
    Features: Dynamic shape.
    Description: Test enable_dynamic and auto dynamic.
    Expectation: No exception.
    """
    generate_dyn("run_enable_auto_dynamic.py", "fn1", "enable_auto_dynamic_shape_fn1.log")


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_arguments_2():
    """
    Features: Dynamic shape.
    Description: Test enable_dynamic and auto dynamic.
    Expectation: No exception.
    """
    generate_dyn("run_enable_auto_dynamic.py", "fn2", "enable_auto_dynamic_shape_fn2.log")
