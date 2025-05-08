# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from functools import lru_cache, wraps
import inspect
import os

import numpy as np


class OpsBinaryCase:
    def __init__(self, input_info, output_info, extra_info='', is_parallel=False):
        if not isinstance(input_info, list) or not isinstance(output_info, list):
            raise TypeError(f'In OpsBinaryCase, input_info and output_info must be list, but got ' \
                            f'input_info({type(input_info)}), output_info({type(output_info)}).')
        self.input_info = input_info
        self.output_info = output_info
        self.extra_info = extra_info
        self.is_parallel = is_parallel
        self.check_info()

    def check_info(self):
        def check_info_valid(info):
            for item in info:
                if not isinstance(item, tuple):
                    raise TypeError(f'For OpsBinaryCase, info must be a tuple containing shape and dtype.')
                if len(item) != 2:
                    raise TypeError(f'For OpsBinaryCase, info must be a tuple containing shape and dtype.')

        if not isinstance(self.input_info, list) or not isinstance(self.output_info, list):
            raise TypeError(f'For OpsBinaryCase, input_info and output_info must be list, ' \
                            f'but got {type(self.input_info)} and {type(self.output_info)}.')
        check_info_valid(self.input_info)
        check_info_valid(self.output_info)

    def check_ops_case(self, input_binary_data, output_binary_data):
        if not isinstance(input_binary_data, list) or not isinstance(output_binary_data, list):
            raise TypeError(f'input_binary_data and output_binary_data must be list, ' \
                            f'but got {type(input_binary_data)} and {type(output_binary_data)}.')
        if len(input_binary_data) != len(self.input_info):
            raise ValueError(f'input binary data files are not enough, but got {len(input_binary_data)} ' \
                             f'while {len(self.input_info)} are needed.')
        if len(output_binary_data) != len(self.output_info):
            raise ValueError(f'output binary data files are not enough, but got {len(output_binary_data)} ' \
                             f'while {len(self.output_info)} are needed.')
        for idx, data in enumerate(input_binary_data):
            if data.shape != self.input_info[idx][0] or data.dtype != self.input_info[idx][1]:
                raise ValueError(f'shape or dtype of input_binary_data[{idx}] is not equal to input_info[{idx}], ' \
                                 f'but got binary shape {data.shape} and input_info shape {self.input_info[idx][0]}, ' \
                                 f'and got binary dtype {data.dtype} and input_info dtype {self.input_info[idx][1]}.')
        for idx, data in enumerate(output_binary_data):
            if data.shape != self.output_info[idx][0] or data.dtype != self.output_info[idx][1]:
                raise ValueError(f'shape or dtype of output_binary_data[{idx}] is not equal to output_info[{idx}], ' \
                                 f'but got binary shape {data.shape} and out_info shape {self.output_info[idx][0]}, ' \
                                 f'and got binary dtype {data.dtype} and out_info dtype {self.output_info[idx][1]}.')


def ops_binary_cases(ops_case, *, binary_data_path=None, debug_info=False):
    if ops_case is None or not isinstance(ops_case, OpsBinaryCase):
        raise TypeError(f'ops_case is invalid.')
    # If you want to run in local environment, please use your own local path
    if binary_data_path is None:
        binary_data_path = '/home/workspace/mindspore_dataset/mindspore-tests-benchmark'

    def decorator(fn):
        if fn is None:
            raise ValueError('fn is None in ops_binary_cases.')

        @wraps(fn)
        def wrapper(*args, **kwargs):
            frame = inspect.currentframe().f_back
            file_path = frame.f_code.co_filename
            work_path, _ = os.path.realpath(__file__).split('/st/ops/test_tools/ops_binary_cases.py')
            relative_path = os.path.relpath(file_path, work_path)
            relative_cases_path, _ = os.path.splitext(relative_path)
            cases_path = os.path.join(binary_data_path, relative_cases_path)
            if debug_info:
                print(f'ops_binary_cases got cases_path: {cases_path}')
            input_binary_data, output_binary_data = ops_binary_cases_read_data(ops_case, cases_path, fn.__name__,
                                                                               debug_info)
            return fn(input_binary_data, output_binary_data, *args, **kwargs)

        return wrapper

    return decorator


@lru_cache(maxsize=10)
def ops_binary_cases_read_data(ops_case, cases_path, case_name, debug_info):
    input_files = []
    output_files = []
    input_case_name = case_name + '_input'
    output_case_name = case_name + '_output'
    for xpath, _, filenames in os.walk(cases_path):
        for filename in filenames:
            full_path = os.path.join(xpath, filename)
            if input_case_name in filename:
                input_files.append(full_path)
            if output_case_name in filename:
                output_files.append(full_path)
    if not input_files:
        raise RuntimeError(f'Can not find valid input data file in {cases_path}.')
    if not output_files:
        raise RuntimeError(f'Can not find valid output data file in {cases_path}.')
    input_files.sort()
    output_files.sort()
    if debug_info:
        for idx, file_path in enumerate(input_files):
            print(f'ops_binary_cases got input_files[{idx}]: {file_path}')
        for idx, file_path in enumerate(output_files):
            print(f'ops_binary_cases got output_files[{idx}]: {file_path}')
    input_binary_data = read_file_by_list(input_files)
    output_binary_data = read_file_by_list(output_files)
    if ops_case.is_parallel:
        rank = int(os.environ.get('RANK_ID'))
        input_binary_data = input_binary_data[rank * len(ops_case.input_info):(rank + 1) * len(ops_case.input_info)]
        output_binary_data = output_binary_data[rank * len(ops_case.output_info):(rank + 1) * len(ops_case.output_info)]
    ops_case.check_ops_case(input_binary_data, output_binary_data)
    return input_binary_data, output_binary_data


def read_file_by_list(files_list):
    load_data = []
    for file in files_list:
        data = np.load(file)
        load_data.append(data)
    return load_data
