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
"""test pijit log"""
import os
import subprocess
from tests.mark_utils import arg_mark


def run_test_case(test_case_name, log_config):
    cmd = f"export MS_JIT_BYTECODE_LOGS={log_config};" \
          + f"python basic_function.py > {test_case_name}_log.txt 2>&1"
    subprocess.check_output(cmd, shell=True)
    data = open(f"{test_case_name}_log.txt", "r").read()
    return data

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_pijit_log_guard():
    """
    Feature: Test pijit log
    Description: Test different pijit log config.
    Expectation: expect to get correct log info.
    """
    test_case_name = "test_pijit_log_guard"
    log_config = "guard"
    data = run_test_case(test_case_name, log_config)

    assert "generated guard at" in data

    os.remove(f"{test_case_name}_log.txt")

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_pijit_log_bytecode():
    """
    Feature: Test pijit log
    Description: Test different pijit log config.
    Expectation: expect to get correct log info.
    """
    test_case_name = "test_pijit_log_bytecode"
    log_config = "bytecode"
    data = run_test_case(test_case_name, log_config)

    assert "ORIGINAL BYTECODE of" in data

    os.remove(f"{test_case_name}_log.txt")

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_pijit_log_graph_break():
    """
    Feature: Test pijit log
    Description: Test different pijit log config.
    Expectation: expect to get correct log info.
    """
    test_case_name = "test_pijit_log_graph_break"
    log_config = "graph_break"
    data = run_test_case(test_case_name, log_config)

    assert "UD analyze: enter GetAliveNodes" in data

    os.remove(f"{test_case_name}_log.txt")

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_pijit_log_all():
    """
    Feature: Test pijit log
    Description: Test different pijit log config.
    Expectation: expect to get correct log info.
    """
    test_case_name = "test_pijit_log_all"
    log_config = "all"
    data = run_test_case(test_case_name, log_config)

    assert "generated guard at" in data
    assert "ORIGINAL BYTECODE of" in data
    assert "UD analyze: enter GetAliveNodes" in data

    os.remove(f"{test_case_name}_log.txt")

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_pijit_log_mix_guard_bytecode():
    """
    Feature: Test pijit log
    Description: Test different pijit log config.
    Expectation: expect to get correct log info.
    """
    test_case_name = "test_pijit_log_mix_guard_bytecode"
    log_config = "guard,bytecode"
    data = run_test_case(test_case_name, log_config)

    assert "generated guard at" in data
    assert "ORIGINAL BYTECODE of" in data
    assert "UD analyze: enter GetAliveNodes" not in data

    os.remove(f"{test_case_name}_log.txt")

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_pijit_log_mix_guard_bytecode_2():
    """
    Feature: Test pijit log
    Description: Test different pijit log config.
    Expectation: expect to get correct log info.
    """
    test_case_name = "test_pijit_log_mix_guard_bytecode_2"
    log_config = "\" guard , bytecode \""
    data = run_test_case(test_case_name, log_config)

    assert "generated guard at" in data
    assert "ORIGINAL BYTECODE of" in data
    assert "UD analyze: enter GetAliveNodes" not in data

    os.remove(f"{test_case_name}_log.txt")

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_pijit_log_mix_guard_all():
    """
    Feature: Test pijit log
    Description: Test different pijit log config.
    Expectation: expect to get correct log info.
    """
    test_case_name = "test_pijit_log_mix_guard_all"
    log_config = "guard,all"
    data = run_test_case(test_case_name, log_config)

    assert "generated guard at" in data
    assert "ORIGINAL BYTECODE of" in data
    assert "UD analyze: enter GetAliveNodes" in data

    os.remove(f"{test_case_name}_log.txt")
