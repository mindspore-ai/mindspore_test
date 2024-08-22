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
import math
import glob
import csv
import mindspore.context as context
import tempfile
import time
import json

from mindspore import Tensor, nn
from pathlib import Path
from tests.mark_utils import arg_mark


class Net(nn.Cell):
    def construct(self, x, y):
        return x + y


def generate_e2edump_json(dump_path, json_file_name, extra_settings_func=None):
    current_dir = Path(__file__).parent
    json_path = current_dir / "test_e2e_statistic_config.json"
    with open(json_path, 'r') as file:
        data = json.load(file)
        data["common_dump_settings"]["path"] = dump_path
        if extra_settings_func is not None:
            extra_settings_func(data)
    with open(json_file_name, 'w') as f:
        json.dump(data, f)


def is_float_equal(value1, value2, rel_tol=1e-4, abs_tol=1e-4):
    try:
        value1 = float(value1)
        value2 = float(value2)
        if math.isnan(value1) and math.isnan(value2):
            return True
        return math.isclose(value1, value2, rel_tol=rel_tol, abs_tol=abs_tol)
    except ValueError:
        return value1 == value2


def to_comparable_pairs(data):
    for key, value in data.items():
        if key in {'Max Value', 'Min Value', 'L2norm Value', 'Avg Value'}:
            yield key, float(value)
        else:
            yield key, value


def match_dicts(target, data):
    for key, target_value in target.items():
        data_value = data.get(key)
        if isinstance(target_value, float):
            if not is_float_equal(target_value, data_value):
                return False
        else:
            if target_value != data_value:
                return False
    return True


def check_statistic_result(data_list, target_list):
    for target in target_list:
        target_pairs = dict(to_comparable_pairs(target))
        assert any(match_dicts(target_pairs, dict(to_comparable_pairs(data)))
                   for data in data_list)


def get_dumped_stat_list(dump_file_path):
    output_name = "statistic.csv"
    output_path = glob.glob(os.path.join(dump_file_path, output_name))[0]
    real_path = os.path.realpath(output_path)
    with open(real_path) as f:
        reader = csv.DictReader(f)
        stats_list = list(reader)
        for stat in stats_list:
            stat.pop(None, None)
        return stats_list


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_e2e_async_statistic_device():
    """
    Feature: kbyk statistic dump support device
    Description: Test kbyk statistic dump on device.
    Expectation: The statistics result meet the requirement.
    """
    context.set_context(mode=context.GRAPH_MODE,
                        device_target="Ascend", jit_config={"jit_level": "O0"})

    def extra_json_settings(data):
        data["e2e_dump_settings"]["stat_calc_mode"] = "device"
        data["e2e_dump_settings"]["enable"] = False
        data["common_dump_settings"]["saved_data"] = "statistic"
        data["common_dump_settings"]["statistic_category"] = [
            "max", "min", "avg", "l2norm"]

    def check_inf_dump(dump_file_path, step):
        stat_list = get_dumped_stat_list(dump_file_path)
        assert len(stat_list) == 3
        common_res = {'Op Type': 'Add', 'Data Size': '12',
                      'Data Type': 'float32', 'Shape': '(3)'}
        target_input0 = {**common_res, **{'IO': 'input', 'Slot': '0'}}
        target_input1 = {**common_res, **{'IO': 'input', 'Slot': '1'}}
        target_output = {**common_res, **{'IO': 'output', 'Slot': '0'}}
        if step == 0:
            target_input0.update(
                {'Max Value': 'nan', 'Min Value': 'nan', 'Avg Value': 'nan', 'L2Norm Value': 'nan'})
            target_input1.update(
                {'Max Value': '2', 'Min Value': '-inf', 'L2Norm Value': 'inf'})
            target_output.update(
                {'Max Value': 'nan', 'Min Value': 'nan', 'Avg Value': 'nan', 'L2Norm Value': 'nan'})
        elif step == 1:
            target_input0.update(
                {'Max Value': 'inf', 'Min Value': '1', 'Avg Value': 'inf', 'L2Norm Value': 'inf'})
            target_input1.update(
                {'Max Value': '2', 'Min Value': '-inf', 'L2Norm Value': 'inf'})
            target_output.update(
                {'Max Value': 'inf', 'Min Value': '-inf', 'Avg Value': 'nan', 'L2Norm Value': 'inf'})
        elif step == 2:
            target_input0.update(
                {'Max Value': '3', 'Min Value': '1', 'Avg Value': '2', 'L2Norm Value': '3.74166'})
            target_input1.update(
                {'Max Value': '2', 'Min Value': '-10', 'Avg Value': '-2', 'L2Norm Value': '10.3923'})
            target_output.update(
                {'Max Value': '4', 'Min Value': '-7', 'Avg Value': '0', 'L2Norm Value': '8.60233'})
        check_statistic_result(
            stat_list, [target_input0, target_input1, target_output])

    with tempfile.TemporaryDirectory(suffix="e2e_statistic_host_with_nan_and_inf") as test_dir:
        path = Path(test_dir)
        dump_path = str(path / "dump_data")
        dump_config_path = str(path / "config.json")
        generate_e2edump_json(dump_path, dump_config_path, extra_json_settings)

        try:
            os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
            net = Net()
            x = Tensor([1., 2., float('nan')])
            y = Tensor([-float("inf"), 2., -10.])
            t1 = net(x, y)
            x = Tensor([1., 2., float('inf')])
            y = Tensor([-float("inf"), 2., -10.])
            t2 = net(x, y)
            x = Tensor([1., 2., 3.])
            y = Tensor([2., 2., -10.])
            t3 = net(x, y)
            print(t1, t2, t3)
            time.sleep(2)
            check_inf_dump(Path(dump_path) / "rank_0" / "Net" / "0" / "0", 0)
            check_inf_dump(Path(dump_path) / "rank_0" / "Net" / "0" / "1", 1)
            check_inf_dump(Path(dump_path) / "rank_0" / "Net" / "0" / "2", 2)
        finally:
            del os.environ['MINDSPORE_DUMP_CONFIG']


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_e2e_sync_statistic_device():
    """
    Feature: kbyk statistic dump support device
    Description: Test kbyk statistic dump on device.
    Expectation: The statistics result meet the requirement.
    """
    context.set_context(mode=context.GRAPH_MODE,
                        device_target="Ascend", jit_config={"jit_level": "O0"})

    def extra_json_settings(data):
        data["e2e_dump_settings"]["stat_calc_mode"] = "device"
        data["common_dump_settings"]["saved_data"] = "statistic"
        data["common_dump_settings"]["statistic_category"] = [
            "max", "min", "avg", "l2norm"]

    def check_inf_dump(dump_file_path, step):
        stat_list = get_dumped_stat_list(dump_file_path)
        assert len(stat_list) == 3
        common_res = {'Op Type': 'Add', 'Data Size': '12',
                      'Data Type': 'float32', 'Shape': '(3)'}
        target_input0 = {**common_res, **{'IO': 'input', 'Slot': '0'}}
        target_input1 = {**common_res, **{'IO': 'input', 'Slot': '1'}}
        target_output = {**common_res, **{'IO': 'output', 'Slot': '0'}}
        if step == 0:
            target_input0.update(
                {'Max Value': 'nan', 'Min Value': 'nan', 'Avg Value': 'nan', 'L2Norm Value': 'nan'})
            target_input1.update(
                {'Max Value': '2', 'Min Value': '-inf', 'L2Norm Value': 'inf'})
            target_output.update(
                {'Max Value': 'nan', 'Min Value': 'nan', 'Avg Value': 'nan', 'L2Norm Value': 'nan'})
        elif step == 1:
            target_input0.update(
                {'Max Value': 'inf', 'Min Value': '1', 'Avg Value': 'inf', 'L2Norm Value': 'inf'})
            target_input1.update(
                {'Max Value': '2', 'Min Value': '-inf', 'L2Norm Value': 'inf'})
            target_output.update(
                {'Max Value': 'inf', 'Min Value': '-inf', 'Avg Value': 'nan', 'L2Norm Value': 'inf'})
        elif step == 2:
            target_input0.update(
                {'Max Value': '3', 'Min Value': '1', 'Avg Value': '2', 'L2Norm Value': '3.74166'})
            target_input1.update(
                {'Max Value': '2', 'Min Value': '-10', 'Avg Value': '-2', 'L2Norm Value': '10.3923'})
            target_output.update(
                {'Max Value': '4', 'Min Value': '-7', 'Avg Value': '0', 'L2Norm Value': '8.60233'})
        check_statistic_result(
            stat_list, [target_input0, target_input1, target_output])

    with tempfile.TemporaryDirectory(suffix="e2e_statistic_host_with_nan_and_inf") as test_dir:
        path = Path(test_dir)
        dump_path = str(path / "dump_data")
        dump_config_path = str(path / "config.json")
        generate_e2edump_json(dump_path, dump_config_path, extra_json_settings)

        try:
            os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
            net = Net()
            x = Tensor([1., 2., float('nan')])
            y = Tensor([-float("inf"), 2., -10.])
            t1 = net(x, y)
            x = Tensor([1., 2., float('inf')])
            y = Tensor([-float("inf"), 2., -10.])
            t2 = net(x, y)
            x = Tensor([1., 2., 3.])
            y = Tensor([2., 2., -10.])
            t3 = net(x, y)
            print(t1, t2, t3)
            time.sleep(2)
            check_inf_dump(Path(dump_path) / "rank_0" / "Net" / "0" / "0", 0)
            check_inf_dump(Path(dump_path) / "rank_0" / "Net" / "0" / "1", 1)
            check_inf_dump(Path(dump_path) / "rank_0" / "Net" / "0" / "2", 2)
        finally:
            del os.environ['MINDSPORE_DUMP_CONFIG']


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_e2e_statistic_host():
    """
    Feature: kbyk statistic dump support host
    Description: Test kbyk statistic dump on host.
    Expectation: The statistics result meet the requirement.
    """
    context.set_context(mode=context.GRAPH_MODE,
                        device_target="Ascend", jit_config={"jit_level": "O0"})

    def extra_json_settings(data):
        data["e2e_dump_settings"]["stat_calc_mode"] = "host"
        data["common_dump_settings"]["saved_data"] = "statistic"
        data["common_dump_settings"]["statistic_category"] = [
            "max", "min", "avg", "l2norm"]

    def check_inf_dump(dump_file_path, step):
        stat_list = get_dumped_stat_list(dump_file_path)
        assert len(stat_list) == 3
        common_res = {'Op Type': 'Add', 'Data Size': '12',
                      'Data Type': 'float32', 'Shape': '(3)'}
        target_input0 = {**common_res, **{'IO': 'input', 'Slot': '0'}}
        target_input1 = {**common_res, **{'IO': 'input', 'Slot': '1'}}
        target_output = {**common_res, **{'IO': 'output', 'Slot': '0'}}
        if step == 0:
            target_input0.update(
                {'Max Value': 'nan', 'Min Value': 'nan', 'Avg Value': 'nan', 'L2Norm Value': 'nan'})
            target_input1.update(
                {'Max Value': '2', 'Min Value': '-inf', 'L2Norm Value': 'inf'})
            target_output.update(
                {'Max Value': 'nan', 'Min Value': 'nan', 'Avg Value': 'nan', 'L2Norm Value': 'nan'})
        elif step == 1:
            target_input0.update(
                {'Max Value': 'inf', 'Min Value': '1', 'Avg Value': 'inf', 'L2Norm Value': 'inf'})
            target_input1.update(
                {'Max Value': '2', 'Min Value': '-inf', 'L2Norm Value': 'inf'})
            target_output.update(
                {'Max Value': 'inf', 'Min Value': '-inf', 'Avg Value': 'nan', 'L2Norm Value': 'inf'})
        elif step == 2:
            target_input0.update(
                {'Max Value': '3', 'Min Value': '1', 'Avg Value': '2', 'L2Norm Value': '3.74166'})
            target_input1.update(
                {'Max Value': '2', 'Min Value': '-10', 'Avg Value': '-2', 'L2Norm Value': '10.3923'})
            target_output.update(
                {'Max Value': '4', 'Min Value': '-7', 'Avg Value': '0', 'L2Norm Value': '8.60233'})
        check_statistic_result(
            stat_list, [target_input0, target_input1, target_output])

    with tempfile.TemporaryDirectory(suffix="e2e_statistic_host_with_nan_and_inf") as test_dir:
        path = Path(test_dir)
        dump_path = str(path / "dump_data")
        dump_config_path = str(path / "config.json")
        generate_e2edump_json(dump_path, dump_config_path, extra_json_settings)

        try:
            os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
            net = Net()
            x = Tensor([1., 2., float('nan')])
            y = Tensor([-float("inf"), 2., -10.])
            t1 = net(x, y)
            x = Tensor([1., 2., float('inf')])
            y = Tensor([-float("inf"), 2., -10.])
            t2 = net(x, y)
            x = Tensor([1., 2., 3.])
            y = Tensor([2., 2., -10.])
            t3 = net(x, y)
            print(t1, t2, t3)
            time.sleep(2)
            check_inf_dump(Path(dump_path) / "rank_0" / "Net" / "0" / "0", 0)
            check_inf_dump(Path(dump_path) / "rank_0" / "Net" / "0" / "1", 1)
            check_inf_dump(Path(dump_path) / "rank_0" / "Net" / "0" / "2", 2)
        finally:
            del os.environ['MINDSPORE_DUMP_CONFIG']
