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
import shutil
from tests.st.networks import utils
from tests.mark_utils import arg_mark


def run_compile_cache_mp(file_name, cache_path, log_file_name_first, log_file_name_second):
    exec_path = os.path.dirname(os.path.realpath(__file__))
    file_name = os.path.join(exec_path, file_name)
    temp_dir = os.path.join(exec_path, "test_run_compile_cache_mp")
    cache_path = os.path.join(temp_dir, cache_path)
    log_file_name_first = os.path.join(temp_dir, log_file_name_first)
    log_file_name_second = os.path.join(temp_dir, log_file_name_second)

    # Clear compile cache folder and log files
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    assert not os.path.exists(cache_path)

    # First run without compile cache
    bash_file = os.path.join(exec_path, "run_compile_cache_mp.sh")
    cmd = "bash {} {} {} {}".format(bash_file, file_name, cache_path, log_file_name_first)
    os.system(cmd)
    check_cmd = "ps -ef | grep python | grep run_compile_cache_mp.py | grep -v grep"

    # wait for net train finish
    ret = utils.process_check(150, check_cmd)
    print("check first train.", flush=True)
    assert ret
    print("check cache file.", flush=True)
    assert os.path.exists(cache_path)

    # First run log file
    from mindspore.communication.management import get_rank
    first_cur_rank_log = os.path.join(log_file_name_first, f'worker_{get_rank()}.log')
    print("check first log.", flush=True)
    assert os.path.exists(first_cur_rank_log)
    with open(first_cur_rank_log, "r") as f_first:
        data_first = f_first.read()
    print("check first compile result.", flush=True)
    assert "Check the consistency of dependency files hash failed. Execute all the compilation actions." in data_first

    # check param status
    no_init = "Before train, the param is inited in parallel_mode, it is False." in data_first
    is_init = "After train, the param is inited in parallel_mode, it is True." in data_first
    assert no_init
    assert is_init

    # Second run
    cmd = "bash {} {} {} {}".format(bash_file, file_name, cache_path, log_file_name_second)
    os.system(cmd)
    ret = utils.process_check(150, check_cmd)
    print("check second train.", flush=True)
    assert ret

    # Second run log file
    second_cur_rank_log = os.path.join(log_file_name_second, f'worker_{get_rank()}.log')
    print("check second log.", flush=True)
    assert os.path.exists(second_cur_rank_log)
    with open(second_cur_rank_log, "r") as f_second:
        data_second = f_second.read()

    has_log = "Use the compilation cache and execute the backend actions only. Be aware of correctness risks." in \
              data_second
    if not has_log:
        print(f'{data_second}')
    print("check second train result.", flush=True)
    assert has_log

    # Clean files
    shutil.rmtree(temp_dir)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='allcards', essential_mark='essential')
def test_compile_cache_pipeline_parallel_and_recompute():
    """
    Feature: Compile cache.
    Description: Test whether pipeline parallel and recompute can successfullty with compile cache.
    Expectation: success.
    """
    run_compile_cache_mp("run_compile_cache_mp.py::test_compile_cache_in_parallel_mode", "./pp_recompute", \
                         "pp_recompute_first", "pp_recompute_second")
