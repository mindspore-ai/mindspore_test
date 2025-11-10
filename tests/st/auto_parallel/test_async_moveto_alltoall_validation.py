"""
Copyright 2025 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import subprocess
import mindspore as ms
from mindspore import context
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_non_blocking_moveto_check_exec_order():
    """
    Feature: Async MoveTo for AlltoAllV sl and rl.
    Description: Check execution order for non-blocking MoveTo.
    Expectation: Raise Exception for wrong execution order.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # run test
    context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", jit_level='O0')
    os.environ['GLOG_v'] = str(1)
    ret = os.system(
        "msrun --worker_num=2 --local_worker_num=2 --master_addr=127.0.0.1 --master_port=51127 --join=True "
        "--log_dir=./test_non_blocking_moveto_check_exec_order pytest -s --disable-warnings "
        "async_moveto_alltoall_validation.py::test_non_blocking_moveto_check_exec_order"
    )
    assert ret == 0

    # clean workspace
    os.system("rm -rf test_non_blocking_moveto_check_exec_order")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_non_blocking_moveto_check_exec_order_with_depend():
    """
    Feature: Async MoveTo for AlltoAllV sl and rl.
    Description: Check execution order for non-blocking MoveTo.
    Expectation: Run successfully.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # run test
    context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", jit_level='O0')
    os.environ['GLOG_v'] = str(0)
    log_file_path = "./non_blocking_moveto_check_exec_order_with_depend"
    ret = os.system(
        "msrun --worker_num=2 --local_worker_num=2 --master_addr=127.0.0.1 --master_port=51128 --join=True "
        f"--log_dir={log_file_path} pytest -s --disable-warnings "
        "async_moveto_alltoall_validation.py::test_non_blocking_moveto_check_exec_order_with_depend"
    )
    assert ret == 0
    check_log = "is blocked before user"
    output = subprocess.check_output(f"grep -r \"{check_log}\" {log_file_path}/worker_0.log | wc -l", shell=True)
    assert int(output) == 2

    # clean workspace
    os.system("rm -rf non_blocking_moveto_check_exec_order_whit_depend")
