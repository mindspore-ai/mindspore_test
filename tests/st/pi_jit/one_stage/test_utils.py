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
"""Test utils for one stage"""
import functools
import glob
import inspect
import os
import shutil
import threading


def count_file_key(file, key):
    """Count key string in file"""
    appear_count = 0
    with open(file, 'r') as fp:
        for line in fp:
            if key in line:
                appear_count += 1
    return appear_count


def check_ir_info(func, inputs, expect_dict, expect_file, expect_num, target_dir):
    """After func run with input, check whether create expect_num of expect_file in target_dir match expect_dict"""
    os.environ['MS_DEV_SAVE_GRAPHS'] = '1'
    os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = target_dir
    os.environ['MS_DEV_DUMP_IR_PASSES'] = expect_file
    try:
        func(*inputs)
        ir_files = glob.glob(os.path.join(target_dir, "*" + expect_file + "*.ir"))
        assert len(ir_files) == expect_num
        for key in expect_dict:
            real_count = 0
            for file in ir_files:
                real_count += count_file_key(file, key)
            assert real_count == expect_dict[key]
    finally:
        os.unsetenv('MS_DEV_SAVE_GRAPHS')
        os.unsetenv('MS_DEV_SAVE_GRAPHS_PATH')
        os.unsetenv('MS_DEV_DUMP_IR_PASSES')
        shutil.rmtree(target_dir)


def _get_ir_path(func) -> str:
    filepath = os.path.basename(inspect.getfile(func))
    filename = os.path.splitext(filepath)[0]
    func_name = func.__name__
    return os.path.join('tmp_ir', filename + '__' + func_name)


IR_PATH_DATA = threading.local()


def _set_current_ir_path(ir_path: str):
    IR_PATH_DATA.ir_path = ir_path


def _get_current_ir_path() -> str:
    return IR_PATH_DATA.ir_path


# A decorator to save IR files automatically.
def save_graph_ir(fn=None, target_dir='AUTO', ir_name='ALL'):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            ir_path = _get_ir_path(func) if target_dir == 'AUTO' else target_dir
            try:
                os.environ['MS_DEV_SAVE_GRAPHS'] = '1'
                os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = ir_path
                if ir_name and ir_name != 'ALL':
                    os.environ['MS_DEV_DUMP_IR_PASSES'] = ir_name
                _set_current_ir_path(ir_path)
                if ir_path and os.path.exists(ir_path):
                    shutil.rmtree(ir_path)
                return func(*args, **kwargs)
            finally:
                os.unsetenv('MS_DEV_SAVE_GRAPHS')
                os.unsetenv('MS_DEV_SAVE_GRAPHS_PATH')
                os.unsetenv('MS_DEV_DUMP_IR_PASSES')
                if ir_path and os.path.exists(ir_path):
                    shutil.rmtree(ir_path)

        return wrapper

    if fn is not None:
        return decorator(fn)
    else:
        return decorator


def check_ir_num(ir_name: str, expect_num: int, target_dir='AUTO'):
    if target_dir == 'AUTO':
        target_dir = _get_current_ir_path()
    if not target_dir:
        assert False, 'Argument target_dir is empty'
    if not os.path.exists(target_dir):
        assert False, 'IR path not exists: ' + target_dir
    ir_files = glob.glob(os.path.join(target_dir, '*' + ir_name + '*.ir'))
    assert len(ir_files) == expect_num, f'IR file num of {ir_name}, expect: {expect_num} vs actual: {len(ir_files)}'
