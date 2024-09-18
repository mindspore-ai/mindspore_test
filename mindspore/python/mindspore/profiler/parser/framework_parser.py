# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""The parser for parsing framework files."""
import glob
import json
import os
import stat
from collections import defaultdict
from collections import namedtuple

import numpy as np
from mindspore.profiler.common.validator.validate_path import validate_and_normalize_path
from mindspore.profiler.parser.framework_enum import FileDataType
from mindspore.profiler.parser.framework_struct import TASK_DESC_STRUCT, TENSOR_DATA_STRUCT, STEP_INFO_STRUCT

FILE_DATA_STRUCT_DICT = {
    FileDataType.STEP_INFO.value: STEP_INFO_STRUCT,
    FileDataType.TENSOR_DATA_INFO.value: TENSOR_DATA_STRUCT,
    FileDataType.TASK_DESC_INFO.value: TASK_DESC_STRUCT
}

TASK_TYPE_TO_KERNEL_TYPE = {
    0: 'AI_CORE',
    1: 'AI_CPU',
    2: 'MSPROF_AIV',
    10: 'MSPROF_HCCL',
    11: 'MSPROF_RTS',
    1000: 'MSPROF_UNKNOWN_TYPE'
}

COL_NAMES = [
    'task_id', 'stream_id', 'block_dim', 'full_op_name', 'op_name', 'op_type', 'subgraph', 'op_info',
    'graph_id', 'kernel_type'
]
OpData = namedtuple('OpData', field_names=COL_NAMES)


class GpuFrameWorkParser:
    """
    The parser for parsing framework files.

    Args:
        output_path (str): The profiling path which should contain GPU profiling data.
        dev_id (str): The device ID.
    """

    _STEPS_TID = 100000
    _GPU_OP_TID = 100002

    def __init__(self, output_path, dev_id, op_names=None):
        """The parser for parsing framework files."""
        self._dev_id = dev_id
        self._output_path = output_path
        self.op_names = op_names
        self.op_name = ''
        self.framework_list = []
        self.op_detail = {}
        self.operation_info = {}
        self.activity_info_dir = []
        self.framework_info_dir = []
        self.cpu_detail_info_dir = []
        self.gpu_op_type_info_dir = []
        self.op_execute_times = {}
        self.op_step_shape_info = defaultdict(list)
        self.one_step_op_time = dict()
        self.one_step_kernel_time = dict()

    def parse(self):
        """Parse op performance data."""
        self.get_device_target_filename()
        self.get_framework_summary()
        self.get_cpu_op_detail_info()
        self.get_activity_op_info()
        if isinstance(self.op_names, str):
            self.combine_performance_data(self.op_names)
        elif isinstance(self.op_names, list):
            for op_name in self.op_names:
                self.combine_performance_data(op_name)
        self.operation_info["device_id"] = self._dev_id
        return json.dumps(self.operation_info)

    def get_framework_summary(self):
        """Get framework data."""
        for filename in self.framework_info_dir:
            op_side = filename.split('_')[0]
            framework_file_path = os.path.join(self._output_path, filename)
            framework_file_path = validate_and_normalize_path(framework_file_path)
            with open(framework_file_path, 'r') as f_obj:
                framework_info = f_obj.readlines()
            for line_info in framework_info:
                line_info = line_info.strip(' ').strip('\n').split(';')
                # line_info[0]: op_type, line_info[1]: op_name, line_info[2]: graph_id, line_info[3]: input_shape;
                input_shape = line_info[3:]
                item = [line_info[0], line_info[1], input_shape, op_side]
                if not self.op_step_shape_info.get(line_info[1]):
                    self.op_step_shape_info[line_info[1]].append(op_side)
                self.op_step_shape_info[line_info[1]].append(input_shape)
                if item not in self.framework_list:
                    self.framework_list.append(item)

    def get_cpu_op_detail_info(self):
        """Get cpu operators detail data."""
        for filename in self.cpu_detail_info_dir:
            op_side = filename.split('_')[0]
            op_detail_file_path = os.path.join(self._output_path, filename)
            op_detail_file_path = validate_and_normalize_path(op_detail_file_path)
            with open(op_detail_file_path, 'r') as f_obj:
                op_detail_info = f_obj.readlines()
            for line_info in op_detail_info[1:]:
                line_info = line_info.strip(' ').strip('\n').split(',')
                if not self.op_detail.get(line_info[2]):
                    # line_info[4]: op_occurrences, line_info[5]: op_detail_time(us), line_info[6]: op_avg_time(us);
                    self.op_detail[line_info[2]] = [float(line_info[4]), float(line_info[5]),
                                                    float(line_info[6]), op_side]

    def get_execute_times(self):
        """Get gpu operators execute times."""
        if self.gpu_op_type_info_dir:
            gpu_op_type_file_path = os.path.join(self._output_path, self.gpu_op_type_info_dir[0])
            gpu_op_type_file_path = validate_and_normalize_path(gpu_op_type_file_path)
            with open(gpu_op_type_file_path, 'r') as fp:
                op_type_info = fp.readlines()
                for line_info in op_type_info[1:]:
                    line_info = line_info.strip(' ').strip('\n').split(',')
                    self.op_execute_times[line_info[0]] = line_info[1]

    def get_activity_op_info(self):
        """Get op detail data."""
        all_file = os.listdir(self._output_path)
        for file_name in all_file:
            if file_name.startswith('gpu_op_type') and file_name.endswith(f'{self._dev_id}.csv'):
                self.gpu_op_type_info_dir.append(file_name)
        if not self.gpu_op_type_info_dir and self.activity_info_dir:
            raise RuntimeError(f'The output file <%s> is not found.' % self.gpu_op_type_info_dir)
        self.get_execute_times()
        for filename in self.activity_info_dir:
            op_side = filename.split('_')[0]
            activity_file_path = os.path.join(self._output_path, filename)
            activity_file_path = validate_and_normalize_path(activity_file_path)
            with open(activity_file_path, 'r') as file:
                activity_info = file.readlines()
            for line_info in activity_info[1:]:
                line_info = line_info.strip(' ').strip('\n').replace(', ', ';').split(',')
                op_name = line_info[2].split('/')[-1]
                # op_name: xxx-opx
                op_type = op_name.split('-')[0]
                op_occurrences = int(self.op_execute_times.get(op_type))
                op_total_time = float(line_info[-4])
                if not self.op_detail.get(op_name):
                    # line_info[4]: op_occurrences, line_info[5]: op_detail_time(us), line_info[6]: op_avg_time(us);
                    self.op_detail[op_name] = [
                        op_occurrences, op_total_time,
                        round(op_total_time / op_occurrences, 4), op_side
                    ]
                else:
                    self.op_detail.get(op_name)[1] += op_total_time
                    self.op_detail.get(op_name)[2] = self.op_detail.get(op_name)[1] / self.op_detail.get(op_name)[0]
                    self.op_detail[op_name] = [
                        self.op_detail.get(op_name)[0],
                        round(self.op_detail.get(op_name)[1], 4),
                        round(self.op_detail.get(op_name)[2], 4), op_side
                    ]

    def combine_performance_data(self, op_name):
        """Combine operator detail info with framework info."""
        unique_op_info = []
        op_shape_dict = {}
        operation_info = {}
        factor = 1000  # convert time unit from ms to us.
        for line_info in self.framework_list:
            op_detail = self.op_detail.get(line_info[1])
            if not op_detail:
                continue
            if op_name in line_info and line_info[3] == op_detail[3]:
                op_side = line_info[3]
                op_shape = '[{}]{}'.format(op_side, ','.join(line_info[2]))
                op_occurrences = int(op_detail[0])
                op_total_time = float(op_detail[1])
                op_avg_time = float(op_detail[2])
                if op_shape in op_shape_dict:
                    # Classify according to the operator information of the same shape.
                    op_shape_dict.get(op_shape)[0] += op_occurrences
                    op_shape_dict.get(op_shape)[1] += op_total_time
                    op_shape_dict.get(op_shape)[2] = op_shape_dict.get(op_shape)[1] / op_shape_dict.get(op_shape)[0]
                    op_shape_dict[op_shape] = [
                        op_shape_dict.get(op_shape)[0], round(op_shape_dict.get(op_shape)[1], 4),
                        round(op_shape_dict.get(op_shape)[2], 4), op_side
                    ]
                else:
                    op_shape_dict[op_shape] = [op_occurrences, op_total_time, op_avg_time, op_side]

        for input_shape in op_shape_dict:
            # 0: op_occurrences, 1: op_total_time, 2: op_avg_time, 3: op_side
            operation_info['op_side'] = op_shape_dict.get(input_shape)[3]
            operation_info['input_shape'] = input_shape.strip('[').split(']')[-1]
            operation_info['op_occurrences'] = op_shape_dict.get(input_shape)[0]
            if operation_info.get('op_side') == 'cpu':
                operation_info['op_total_time(us)'] = round(op_shape_dict.get(input_shape)[1] * factor, 4)
                operation_info['op_avg_time(us)'] = round(op_shape_dict.get(input_shape)[2] * factor, 4)
            else:
                operation_info['op_total_time(us)'] = op_shape_dict.get(input_shape)[1]
                operation_info['op_avg_time(us)'] = op_shape_dict.get(input_shape)[2]
            unique_op_info.append(operation_info)
            operation_info = dict()

        if unique_op_info:
            self.operation_info[op_name] = unique_op_info
        else:
            raise RuntimeError(f'The information of <{op_name}> is not found. Please verify that the operator name is'
                               f' correct or the operator is used in the network.')

    def get_device_target_filename(self):
        """Get device target filename."""
        gpu_framework_file = f'gpu_framework_{self._dev_id}.txt'
        cpu_framework_file = f'cpu_framework_{self._dev_id}.txt'
        gpu_activity_file = f'gpu_activity_data_{self._dev_id}.csv'
        cpu_op_detail_file = f'cpu_op_detail_info_{self._dev_id}.csv'
        all_file = os.listdir(self._output_path)
        if not all_file:
            raise RuntimeError(f'No profiler file is found in the path <%s>. '
                               f'Check whether the profiler path is correct.' % self._output_path)
        if gpu_activity_file in all_file and gpu_framework_file not in all_file:
            raise RuntimeError(f'The output file <%s> is not found.' % gpu_framework_file)
        if cpu_op_detail_file in all_file and cpu_framework_file not in all_file:
            raise RuntimeError(f'The output file <%s> is not found.' % cpu_framework_file)
        if gpu_framework_file in all_file and gpu_activity_file not in all_file:
            raise RuntimeError(f'The output file <%s> is not found.' % gpu_activity_file)
        if cpu_framework_file in all_file and cpu_op_detail_file not in all_file:
            raise RuntimeError(f'The output file <%s> is not found.' % cpu_op_detail_file)
        if gpu_activity_file not in all_file and cpu_op_detail_file not in all_file:
            raise RuntimeError(f'The profiling data of this card which device_id is equal to {self._dev_id} does not'
                               f' exist. Check whether device_id is correct.')
        for file_name in all_file:
            if file_name.endswith(f'activity_data_{self._dev_id}.csv'):
                self.activity_info_dir.append(file_name)
            if file_name.endswith(f'framework_{self._dev_id}.txt'):
                self.framework_info_dir.append(file_name)
            if file_name.startswith('cpu_op_detail') and file_name.endswith(f'{self._dev_id}.csv'):
                self.cpu_detail_info_dir.append(file_name)

    def analyse_dynamic_shape_data(self, timeline_meta):
        """Analyse gpu operators's information and cudakernel's information."""
        kernel_info = defaultdict(list)
        operator_info = defaultdict(list)
        kernel_type_step_time = dict()
        op_type_step_time = dict()
        step, first_update = 1, 0
        self.get_device_target_filename()
        self.get_framework_summary()
        for op_info in timeline_meta:
            args = op_info.get("args", {})
            if op_info.get("tid") == self._STEPS_TID and op_info.get('dur'):
                step = int(op_info.get("name"))
                if first_update:
                    self.one_step_op_time = dict()
                    self.one_step_kernel_time = dict()
                first_update = 1
            elif args and args.get("type") == "cuLaunchKernel":
                item = self._organize_result(step, op_info, args)
                kernel_info[step].append(item)
                self._get_one_step_info(item, "kernel")
            elif (op_info.get("tid") == self._GPU_OP_TID and not op_info.get("cat")) or \
                    str(op_info.get("tid")).startswith('HostCpu'):
                item = self._organize_result(step, op_info, args)
                operator_info[step].append(item)
                self._get_one_step_info(item, "operator")
            op_type_step_time[step] = self.one_step_op_time
            kernel_type_step_time[step] = self.one_step_kernel_time
        self.write_dynamic_shape_data(operator_info, kernel_info, op_type_step_time, kernel_type_step_time)

    def write_dynamic_shape_data(self, operator_info, kernel_info, op_type_step_time, kernel_type_step_time):
        """Organize the result."""
        output_dynamic_shape_file_name = f"dynamic_shape_info_{self._dev_id}.json"
        result = {
            "operator": operator_info,
            "kernel": kernel_info,
            "operator_type": op_type_step_time,
            "kernel_type": kernel_type_step_time,
        }
        dynamic_shape_file_path = os.path.join(self._output_path, output_dynamic_shape_file_name)
        with os.fdopen(os.open(dynamic_shape_file_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), 'w') as fp:
            json.dump(result, fp)
        os.chmod(dynamic_shape_file_path, stat.S_IREAD | stat.S_IWRITE)

    def get_graph_ids(self):
        """Get gpu graph ids."""
        gpu_framework_file = list(glob.glob(os.path.join(self._output_path,
                                                         'gpu_framework_{}.txt'.format(self._dev_id))))
        if not gpu_framework_file:
            return []
        graph_ids = set()
        with open(gpu_framework_file[0], 'r') as f_obj:
            framework_info = f_obj.readlines()
        for line_info in framework_info:
            if line_info.startswith("InitDataSetQueue") or line_info.startswith("GetNext"):
                continue
            line_info = line_info.strip(' ').strip('\n').split(';')
            if len(line_info) > 2 and line_info[2].isdigit():
                graph_ids.add(int(line_info[2]))
        return list(graph_ids)

    def _organize_result(self, step, op_info, args):
        """Organize the results."""
        if args.get("type", "") == "cuLaunchKernel":
            item = {
                "step": step,
                "op_type": args.get("type"),
                "op_name": op_info.get('name'),
                "op_full_name": args.get('op_full_name'),
                "dur": op_info.get('dur'),
                "block_dim": args.get('block_dim'),
                "grid_dim": args.get('grid_dim')
            }
        else:
            op_step_shape = self.op_step_shape_info.get(op_info.get('name'))
            item = {
                "step": step,
                "op_side": self.op_step_shape_info.get(op_info.get('name'))[0],
                "op_type": op_info.get('name').split('-')[0],
                "op_name": op_info.get('name'),
                "dur": op_info.get('dur'),
                "shape_info": op_step_shape[step] if len(op_step_shape) > step else [],
            }
        return item

    def _get_one_step_info(self, item, op_type):
        """Get operator type information in step."""
        duration = item.get("dur")
        if op_type == "operator":
            sort_type = item.get("op_type")
            if not self.one_step_op_time.get(sort_type):
                # duration, times, avg_time
                self.one_step_op_time[sort_type] = [duration, 1, duration]
            else:
                self.one_step_op_time[sort_type][0] += duration
                self.one_step_op_time[sort_type][1] += 1
                self.one_step_op_time[sort_type] = [
                    self.one_step_op_time[sort_type][0],
                    self.one_step_op_time[sort_type][1],
                    round(self.one_step_op_time[sort_type][0] /
                          self.one_step_op_time[sort_type][1], 4)
                ]
        else:
            sort_type = item.get("op_name")
            op_full_name = item.get("op_full_name")
            if not self.one_step_kernel_time.get(sort_type):
                # duration, times, avg_time
                self.one_step_kernel_time[sort_type] = [duration, 1, duration, op_full_name]
            else:
                self.one_step_kernel_time[sort_type][0] += duration
                self.one_step_kernel_time[sort_type][1] += 1
                self.one_step_kernel_time[sort_type] = [
                    self.one_step_kernel_time[sort_type][0],
                    self.one_step_kernel_time[sort_type][1],
                    round(self.one_step_kernel_time[sort_type][0] /
                          self.one_step_kernel_time[sort_type][1], 4),
                    op_full_name
                ]


class DynamicFrameWorkParser:
    """
    Thr parser for parsing dynamic shape framework files.

    Args:
        output_path (str): The profiling path which should contain Ascend profiling data.
        rank_id (int): The rank ID.
    """

    def __init__(self, output_path, rank_id, pretty=False):
        """Initialization of parsing framework data."""
        self._output_path = output_path
        self._all_op_exe_time = defaultdict(list)
        self._op_shape_info = defaultdict(list)
        self._op_info = dict()
        self._rank_id = rank_id
        self._op_type_exe_time = defaultdict(list)
        self._exe_time_and_shape_detail = defaultdict(dict)
        self._dynamic_shape_info = defaultdict(list)
        self._step = 0
        self._pretty = pretty

    @property
    def indent(self):
        indent = 1 if self._pretty else None
        return indent

    def write_dynamic_shape_data(self, df_op_summary):
        """Analyze dynamic shape data and write to dynamic shape file."""
        self._get_total_step_num(df_op_summary)
        output_dynamic_shape_file_name = f'dynamic_shape_info_{self._rank_id}.json'
        for op_name in self._exe_time_and_shape_detail:
            if self._exe_time_and_shape_detail[op_name]['op_exe_occurrences'] == self._step:
                self._op_info[op_name] = self._exe_time_and_shape_detail.get(op_name)
        for op_name, op_detail in self._op_info.items():
            op_type = op_name.split('-', maxsplit=1)[0]
            item = {op_name: op_detail}
            self._dynamic_shape_info[op_type].append(item)
        self._op_info["op_type"] = dict()
        for op_name in self._op_info:
            if op_name != 'op_type':
                op_type = op_name.split('-')[0]
                self._op_type_exe_time[op_type].append(self._all_op_exe_time[op_name])
        for op_type in self._op_type_exe_time:
            if self._op_type_exe_time[op_type]:
                self._op_info.get("op_type", {})[op_type] = (
                    np.around(np.sum(self._op_type_exe_time[op_type], axis=0, dtype='float') /
                              len(self._op_type_exe_time[op_type]), 4)).tolist()
        self._dynamic_shape_info['op_type'] = self._op_info.get("op_type")
        dynamic_shape_file_path = os.path.join(self._output_path, output_dynamic_shape_file_name)
        with os.fdopen(os.open(dynamic_shape_file_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), 'w') as fp:
            json.dump(self._dynamic_shape_info, fp, indent=self.indent)
        os.chmod(dynamic_shape_file_path, stat.S_IREAD | stat.S_IWRITE)

    def _analyse_op_execute_time(self, op_summary):
        """Obtain the execution time of aicpu operator and aicore operator."""
        timeline_info = defaultdict(list)
        for row in op_summary:
            key = row['Op Name'].split('/')[-1]
            timeline_info[key].append(row['Task Duration'])

        self._all_op_exe_time = timeline_info

    def _get_dynamic_shape_info(self, op_summary):
        """Get the shape information of AICPU and aicore."""
        framework_file_name = f'framework_raw_{self._rank_id}.csv'
        self._analyse_op_execute_time(op_summary)
        framework_file_path = os.path.join(self._output_path, framework_file_name)
        framework_file_path = validate_and_normalize_path(framework_file_path)
        with open(framework_file_path, 'r') as f_obj:
            framework_info = f_obj.readlines()[1:]
            for line_info in framework_info:
                line_info = line_info.strip('\n').split(',')
                op_name = line_info[3].split('/')[-1]
                shape_info = ','.join(line_info[8:]).replace('"', '')
                self._op_shape_info[op_name].append(shape_info)

    def _get_total_step_num(self, op_summary):
        """Get the number of steps."""
        self._get_dynamic_shape_info(op_summary)
        all_exe_occurrences = list()
        for op_name in self._all_op_exe_time:
            op_shape = self._op_shape_info.get(op_name)
            op_exe_time_list = self._all_op_exe_time.get(op_name)
            if not op_shape:
                continue
            if len(op_shape) == len(op_exe_time_list):
                self._exe_time_and_shape_detail[op_name]['op_exe_time'] = op_exe_time_list
                self._exe_time_and_shape_detail[op_name]['op_shape'] = op_shape
                self._exe_time_and_shape_detail[op_name]['op_exe_occurrences'] = len(op_exe_time_list)
                all_exe_occurrences.append(len(op_exe_time_list))
        self._step = max(set(all_exe_occurrences), key=all_exe_occurrences.count)
