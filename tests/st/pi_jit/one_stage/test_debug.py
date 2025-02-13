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
"""Test one stage debug info"""
import os
import glob
import shutil

from mindspore import jit
from mindspore import ops
from mindspore import Tensor
from tests.mark_utils import arg_mark
from tests.st.pi_jit.share.utils import pi_jit_with_config

def count_file_key(file, key):
    """Count key string in file"""
    appear_count = 0
    with open(file, 'r') as fp:
        for line in fp:
            if key in line:
                appear_count += 1
    return appear_count


def check_debug_info_no_break(func, inputs, expect_dict, dir):
    """Check whether func(inputs) create IR match expect dict"""
    os.environ['MS_DEV_SAVE_GRAPHS'] = '1'
    os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = dir
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'graph_before_compile'
    try:
        func(*inputs)
        ir_files = glob.glob(os.path.join(dir, 'graph_before_compile*.ir'))
        assert len(ir_files) == 1
        file = ir_files[0]
        for key in expect_dict:
            assert count_file_key(file, key) == expect_dict[key]
    finally:
        os.unsetenv('MS_DEV_SAVE_GRAPHS')
        os.unsetenv('MS_DEV_SAVE_GRAPHS_PATH')
        os.unsetenv('MS_DEV_DUMP_IR_PASSES')
        shutil.rmtree(dir)


def check_debug_info_with_break(func, inputs, expect_ir_num, expect_dict, dir):
    os.environ['MS_DEV_SAVE_GRAPHS'] = '1'
    os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = dir
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'graph_before_compile'
    try:
        func(*inputs)
        ir_files = glob.glob(os.path.join(dir, 'graph_before_compile*.ir'))
        assert len(ir_files) == expect_ir_num
        for key in expect_dict:
            real_count = 0
            for file in ir_files:
                real_count += count_file_key(file, key)
            assert real_count == expect_dict[key]
    finally:
        os.unsetenv('MS_DEV_SAVE_GRAPHS')
        os.unsetenv('MS_DEV_SAVE_GRAPHS_PATH')
        os.unsetenv('MS_DEV_DUMP_IR_PASSES')
        shutil.rmtree(dir)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_debug_info_for_binary_op():
    """
    Feature: PIJit stage debug info in IR.
    Description: Test whether debug info is in IR for PIJit.
    Expectation: No exception, the IR should have debug info.
    """

    @pi_jit_with_config(jit_config={"compile_with_try": False})
    def foo(x, y):
        m = x + y
        return m

    input_x = Tensor([1, 2, 3])
    input_y = Tensor([2, 3, 4])
    check_debug_info_no_break(foo, (input_x, input_y), {"m = x + y": 1}, "./test_debug_info_for_binary_op")


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_debug_info_for_primitive_call():
    """
    Feature: PIJit stage debug info in IR.
    Description: Test whether debug info is in IR for PIJit.
    Expectation: No exception, the IR should have debug info.
    """

    @pi_jit_with_config(jit_config={"compile_with_try": False})
    def foo(x, y):
        m = ops.add(x, y)
        return m

    input_x = Tensor([1, 2, 3])
    input_y = Tensor([2, 3, 4])
    check_debug_info_no_break(foo, (input_x, input_y), {"m = ops.add(x, y)": 3}, "./test_debug_info_for_primitive_call")


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_debug_info_for_func_graph_call():
    """
    Feature: PIJit stage debug info in IR.
    Description: Test whether debug info is in IR for PIJit.
    Expectation: No exception, the IR should have debug info.
    """
    def inner(x, y):
        ret = x + y
        return ret

    @pi_jit_with_config(jit_config={"compile_with_try": False})
    def foo(x, y):
        m = inner(x, y)
        return m

    input_x = Tensor([1, 2, 3])
    input_y = Tensor([2, 3, 4])
    expect_dict = {"ret = x + y": 1, "m = inner(x, y)": 3}
    check_debug_info_no_break(foo, (input_x, input_y), expect_dict, "./test_debug_info_for_func_graph_call")


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_debug_info_for_binary_op_with_break():
    """
    Feature: PIJit stage debug info in IR.
    Description: Test whether debug info is in IR for PIJit.
    Expectation: No exception, the IR should have debug info.
    """

    @pi_jit_with_config(jit_config={"compile_with_try": False})
    def foo(x, y):
        m = x + y
        print("aaaa", flush=True)  # break here
        n = x - y
        return m, n

    input_x = Tensor([1, 2, 3])
    input_y = Tensor([2, 3, 4])
    expect_dict = {"m = x + y": 1, "n = x - y": 1, "return m, n": 1}
    check_debug_info_with_break(foo, (input_x, input_y), 2, expect_dict, "./test_debug_info_for_binary_op_with_break")


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_debug_info_for_graph_call_with_break():
    """
    Feature: PIJit stage debug info in IR.
    Description: Test whether debug info is in IR for PIJit.
    Expectation: No exception, the IR should have debug info.
    """
    def inner(x, y):
        ret = x + y
        return ret

    @pi_jit_with_config(jit_config={"compile_with_try": False})
    def foo(x, y):
        m = x + y
        print("aaaa", flush=True)  # break here
        n = inner(x, y)
        return m, n

    input_x = Tensor([1, 2, 3])
    input_y = Tensor([2, 3, 4])
    expect_dict = {"m = x + y": 1, "ret = x + y": 1, "n = inner(x, y)": 3, "return m, n": 1}
    check_debug_info_with_break(foo, (input_x, input_y), 2, expect_dict, "./test_debug_info_for_graph_call_with_break")


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_debug_info_for_ops_call_with_break():
    """
    Feature: PIJit stage debug info in IR.
    Description: Test whether debug info is in IR for PIJit.
    Expectation: No exception, the IR should have debug info.
    """
    def inner(x, y):
        ret = x + y
        return ret

    @pi_jit_with_config(jit_config={"compile_with_try": False})
    def foo(x, y):
        m = x + y
        print("aaaa", flush=True)  # break here
        n = inner(x, y)
        print("bbbb", flush=True)  # break here
        z = ops.add(x, y)
        return m, n, z

    input_x = Tensor([1, 2, 3])
    input_y = Tensor([2, 3, 4])
    expect_dict = {"m = x + y": 1, "ret = x + y": 1, "n = inner(x, y)": 3, "z = ops.add(x, y)": 3, "return m, n": 1}
    check_debug_info_with_break(foo, (input_x, input_y), 3, expect_dict, "./test_debug_info_for_ops_call_with_break")
