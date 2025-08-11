# Copyright 2024-2025 Huawei Technologies Co., Ltd
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
import shutil
import subprocess
from tests.mark_utils import arg_mark


def _delete_dir(dir_name):
    if os.path.isdir(dir_name):
        shutil.rmtree(dir_name)


def _run_new_process(file_name, output_dir):
    _delete_dir(output_dir)
    os.makedirs(output_dir)

    # env
    base_env = "DVM_OUTPUT_DIR={}".format(output_dir)
    # dvm env
    non_default_ops = ["Dense", "MatMul", "MatMulExt", "BatchMatMul", "BatchMatMulExt"]
    dvm_env = 'MS_DEV_PYNATIVE_FUSION_FLAGS="--opt_level=1 --enable_ops={}"'.format(",".join(non_default_ops))
    dvm_env = dvm_env + " " + base_env
    commands = [
        "{} pytest -s {}".format(base_env, file_name),
        "{} pytest -s {}".format(dvm_env, file_name)
    ]
    for command in commands:
        try:
            subprocess.run(command, shell=True, check=True, text=True)
        except Exception as e:
            _delete_dir(output_dir)
            raise e
    _delete_dir(output_dir)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_op():
    """
    Feature: test dvm op precision
    Description: pynative mode
    Expectation: the result match with the expected result
    """
    _run_new_process("dvm_op_pynative.py", "./dvm_op")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_matmul():
    """
    Feature: test dvm matmul op precision
    Description: pynative mode
    Expectation: the result match with the expected result
    """
    _run_new_process("dvm_matmul_pynative.py", "./dvm_matmul")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_fuse():
    """
    Feature: test dvm fuse op precision
    Description: pynative mode
    Expectation: the result match with the expected result
    """
    _run_new_process("dvm_fuse_pynative.py", "./dvm_fuse")
