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

from multiprocessing.pool import Pool
from tests.mark_utils import arg_mark
from tests.st.networks.utils import get_num_from_log

os.environ["MS_SUBMODULE_LOG_v"] = "{DEVICE:1}"
TOELERANCE = 5e-2
PEAK_MEMORY_NAME = "Actual peak memory usage (with fragments):"

cur_dir = os.path.dirname(os.path.abspath(__file__))


def run_command(command_info):
    cmd, log_path, expect_peak_memory = command_info
    ret = os.system(cmd)
    return ret, log_path, expect_peak_memory


def check_results(commands, results):
    error_idx = [_ for _ in range(len(results)) if results[_][0] != 0]
    for idx in error_idx:
        print(f"testcase {commands[idx]} failed. please check log {results[idx][1]}.")
        os.system(f"grep -E 'ERROR|error|Error' {results[idx][1]} -C 5")
    assert error_idx == []

    # check peak_memory
    check_memory = True
    for _, log_path, expect_peak_memory in results:
        print("log_path is:", log_path)
        print("run peak_memory is:", expect_peak_memory)
        peak_memory = get_num_from_log(log_path, PEAK_MEMORY_NAME)
        if peak_memory <= expect_peak_memory * (1 + TOELERANCE):
            continue
        check_memory = False
        print(f"The peak_memory in log {log_path} is {peak_memory}, "
              f"the error between {peak_memory} and standard value {expect_peak_memory} is greater than {TOELERANCE}.")
    if not check_memory:
        raise RuntimeError("run above commands failed, please check error in log.")


class TestInferParallel:
    """A test class for testing pipeline."""

    @staticmethod
    def setup_method():
        ascend_home_path = os.getenv('ASCEND_HOME_PATH')
        if not ascend_home_path:
            os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"

    @arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards', essential_mark='essential')
    def test_base_cases(self):
        """
        Feature: Infer interface
        Description: Test parallel interface for training and prediction.
        Expectation: AssertionError
        """
        commands = [
            (f"export ASCEND_RT_VISIBLE_DEVICES=0,1 && export LCAL_COMM_ID=127.0.0.1:10068 && msrun --worker_num=2 "
             f"--local_worker_num=2 --master_port=8222 --log_dir=parallel_qwen2_0_5b_predict_mp2 --join=True "
             f"{cur_dir}/run_parallel.py --mode parallel_qwen2_0_5b_predict_mp2",
             'parallel_qwen2_0_5b_predict_mp2/worker_0.log', 792),  # command, log_path, expect_peak_memory
            (f"export ASCEND_RT_VISIBLE_DEVICES=2,3 && export LCAL_COMM_ID=127.0.0.1:10070 && msrun --worker_num=2 "
             f"--local_worker_num=2 --master_port=8230 --log_dir=parallel_qwen2_0_5b_predict_mp2_static --join=True  "
             f"{cur_dir}/run_parallel.py --mode parallel_qwen2_0_5b_predict_mp2_static",
             'parallel_qwen2_0_5b_predict_mp2_static/worker_0.log', 868),
            (f"export ASCEND_RT_VISIBLE_DEVICES=4,5 && export LCAL_COMM_ID=127.0.0.1:10072 && msrun --worker_num=2 "
             f"--local_worker_num=2 --master_port=8238 --log_dir=parallel_qwen2_0_5b_parallel_decoding_mp2 --join=True "
             f"{cur_dir}/run_parallel.py --mode parallel_qwen2_0_5b_parallel_decoding_mp2",
             'parallel_qwen2_0_5b_parallel_decoding_mp2/worker_0.log', 810),
            (f"export ASCEND_RT_VISIBLE_DEVICES=6,7 && export LCAL_IF_PORT=10074 && msrun --worker_num=2 "
             f"--local_worker_num=2 --master_port=8246 --log_dir=parallel_qwen2_0_5b_multilora_mp2 --join=True "
             f"{cur_dir}/run_parallel.py --mode parallel_qwen2_0_5b_multilora_mp2",
             'parallel_qwen2_0_5b_multilora_mp2/worker_0.log', 800),
        ]

        with Pool(len(commands)) as pool:
            results = list(pool.imap(run_command, commands))
        check_results(commands, results)
