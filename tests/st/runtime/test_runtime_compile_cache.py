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

import numpy as np
import os
import re
import shutil
import subprocess
import hashlib
from mindspore import context
from tests.mark_utils import arg_mark

context.set_context(device_target="Ascend")
context.set_context(jit_config={"jit_level": "O0"})


match_output = re.compile(r'AAA(.*?)BBB', re.S)
match_num = re.compile(r'\d+\.?\d*', re.S)


def check_log(role, log_name, str_to_check):
    assert os.path.exists(role + "/" + log_name)
    with open(role + "/" + log_name, "r") as f:
        data = f.read()
    assert str_to_check in data


def clear_and_make_run_dir(dir_path):
    shutil.rmtree(dir_path, ignore_errors=True)
    assert not os.path.exists(dir_path)
    os.mkdir(dir_path)
    assert os.path.exists(dir_path)


def check_backend_compile_cache_files(cache_path):
    assert os.path.exists(cache_path)
    # Funcgraph
    assert os.path.exists(cache_path + "/rank_0/graph_cache/compile_dependency.hash")
    assert os.path.exists(cache_path + "/rank_0/graph_cache/compile_cache_0.mindir")
    # Kernelgraph
    assert os.path.exists(cache_path + "/rank_0/graph_cache/backend_compile_cache_0.json")
    assert os.path.exists(cache_path + "/rank_0/graph_cache/backend_compile_cache_0.mindir")


def get_hash_file_md5(cache_path):
    hash_file_path = cache_path + "/rank_0/graph_cache/compile_dependency.hash"
    assert os.path.exists(hash_file_path)
    hash_alg = hashlib.md5()
    with open(hash_file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_alg.update(chunk)
    computed_hash = hash_alg.hexdigest()
    return computed_hash


def check_hash_file(cache_path, first_run_hash):
    assert os.path.exists(cache_path)
    hash_file_path = cache_path + "/rank_0/graph_cache/compile_dependency.hash"
    assert os.path.exists(hash_file_path)
    hash_alg = hashlib.md5()
    with open(hash_file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_alg.update(chunk)
    computed_hash = hash_alg.hexdigest()
    return computed_hash == first_run_hash


def run_twice_with_same_network(file_name, cache_path, log_file_name_first, log_file_name_second):
    # Clear compile cache folder and log files
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)
    if os.path.exists(log_file_name_first):
        os.remove(log_file_name_first)
    if os.path.exists(log_file_name_second):
        os.remove(log_file_name_second)
    assert not os.path.exists(cache_path)
    assert not os.path.exists(log_file_name_first)
    assert not os.path.exists(log_file_name_second)

    # Only support kbk
    assert os.path.exists(file_name)
    temp_file = file_name + ".tp.py"
    with open(file_name, "r") as file:
        ctx = file.read()
    ctx = ctx.replace("O2", "O0")
    with open(temp_file, "w") as file:
        file.write(ctx)
    assert os.path.exists(temp_file)

    os.environ['MS_DEV_RUNTIME_CONF'] = "memory_statistics:True,compile_statistics:True,backend_compile_cache:True"
    # First run without compile cache
    cmd_first = f"export GLOG_v=1; python " + temp_file + " '" \
        + cache_path + "' > " + log_file_name_first + " 2>&1"
    subprocess.check_output(cmd_first, shell=True)
    assert os.path.exists(log_file_name_first)
    assert os.path.exists(cache_path)
    with open(log_file_name_first, "r") as f_first:
        data_first = f_first.read()

    # First run check compile cache end and save compile cache
    assert "Status record: Start cache backend kernel graph." in data_first
    assert "Dump control node cache success." in data_first
    assert "Status record: End cache backend kernel graph." in data_first
    assert data_first.count("[PROF]compile_backend_graph cost") == 2
    assert "Status record: end compile function graph:" in data_first

    # Take out the result of the first run
    match_output_first = re.findall(match_output, data_first)
    assert len(match_output_first) == 2
    nums_first = re.findall(match_num, match_output_first[0])
    array_first = np.array([float(x) for x in nums_first])
    shape_first = re.findall(match_num, match_output_first[1])
    array_shape_first = np.array([int(x) for x in shape_first])

    # Check .mindir .json before second run with compile cache
    check_backend_compile_cache_files(cache_path)
    # Check hash is same before second run with compile cache
    first_run_hash = get_hash_file_md5(cache_path)
    assert check_hash_file(cache_path, first_run_hash)

    # Second run with compile cache
    cmd_second = f"export GLOG_v=1; python " + temp_file + " '" \
        + cache_path + "' > " + log_file_name_second + " 2>&1"
    subprocess.check_output(cmd_second, shell=True)
    assert os.path.exists(log_file_name_second)
    with open(log_file_name_second, "r") as f_second:
        data_second = f_second.read()

    # Second run can compile cache
    assert "Enable backend compile cache." in data_second
    assert "Status record: Start load backend kernel graph." in data_second
    assert "Status record: start use cache to compile graph kbk." in data_second
    assert data_second.count("[PROF]Load_backend_compile_cache") == 2

    # Take out the result of the second run
    match_output_second = re.findall(match_output, data_second)
    assert len(match_output_second) == 2
    nums_second = re.findall(match_num, match_output_second[0])
    array_second = np.array([float(x) for x in nums_second])
    shape_second = re.findall(match_num, match_output_second[1])
    array_shape_second = np.array([int(x) for x in shape_second])

    assert np.allclose(array_first, array_second, 0.0001, 0.0001)
    assert (array_shape_first == array_shape_second).all()

    # Clean files
    os.remove(log_file_name_first)
    os.remove(log_file_name_second)
    os.remove(temp_file)
    shutil.rmtree(cache_path)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_compile_cache_load_weights():
    """
    Feature: compile cache.
    Description: test whether the compile cache can load the value of parameters successfully.
    Expectation: success.
    """
    fpath = os.path.abspath(os.path.dirname(os.getcwd()))
    pypath = fpath + "/compiler/compile_cache/run_network_with_weights.py"
    run_twice_with_same_network(pypath, "./weight", "weight_first.txt", "weight_second.txt")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_compile_cache_lenet():
    """
    Feature: compile cache.
    Description: test whether the regular compile cache function can run successfully.
    Expectation: success.
    """
    fpath = os.path.abspath(os.path.dirname(os.getcwd()))
    pypath = fpath + "/compiler/compile_cache/run_lenet.py"
    run_twice_with_same_network(pypath, "./lenet", "lenet_first.txt", "lenet_second.txt")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_compile_cache_ms_function():
    """
    Feature: compile cache.
    Description: test whether the compile cache function can run successfully in the compilation of ms_function.
    Expectation: success.
    """
    fpath = os.path.abspath(os.path.dirname(os.getcwd()))
    pypath = fpath + "/compiler/compile_cache/run_lenet_ms_function.py"
    run_twice_with_same_network(pypath, "./lenet_ms_function", "lenet_ms_function_first.txt",
                                "lenet_ms_function_second.txt")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_compile_cache_lenet_ge():
    """
    Feature: compile cache.
    Description: Test whether the ge compile cache function can run successfully.
    Expectation: success.
    """
    fpath = os.path.abspath(os.path.dirname(os.getcwd()))
    pypath = fpath + "/compiler/compile_cache/run_lenet.py"
    run_twice_with_same_network(pypath, "./lenet", "lenet_first.txt", "lenet_second.txt")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_resnet_infer_compile_cache():
    """
    Feature: support compile cache in inference scenarios.
    Description: support compile cache in inference scenarios.
    Expectation: success.
    """
    fpath = os.path.abspath(os.path.dirname(os.getcwd()))
    pypath = fpath + "/compiler/compile_cache/run_resnet_infer.py"
    run_twice_with_same_network(pypath, "./resnet_infer", "resnet_infer_first.txt",
                                "resnet_infer_second.txt")
