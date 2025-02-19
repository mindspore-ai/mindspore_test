# Copyright 2024-2025 Huawei Technologies Co., Ltd
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
"""test dynamic step profiler"""
import glob
import os
import tempfile
import numpy as np
import pandas as pd

import mindspore as ms
import mindspore.profiler as prof
import mindspore.dataset as ds
from mindspore.profiler import Profiler, ProfilerLevel, ProfilerActivity
from mindspore import Tensor, context, nn
from mindspore.profiler.profiler_interface import ProfilerInterface

from file_check import FileChecker
from model_zoo import TinyAddNet
from tests.mark_utils import arg_mark


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Dense(2, 2)

    def construct(self, x):
        return self.fc(x)


def generator_net():
    for _ in range(2):
        yield np.ones([2, 2]).astype(np.float32), np.ones([2]).astype(np.int32)


def train_net(net):
    optimizer = nn.Momentum(net.trainable_params(), 1, 0.9)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    data = ds.GeneratorDataset(generator_net(), ["data", "label"])
    model = ms.train.Model(net, loss, optimizer)
    model.train(1, data)


def train(add):
    """ Train add net"""
    x = np.random.randn(1, 3, 3, 4).astype(np.float32)
    y = np.random.randn(1, 3, 3, 4).astype(np.float32)
    add(Tensor(x), Tensor(y))


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_step_single_active_kbk_profiler():
    """
    Feature: Dynamic Step Profiler
    Description: This test case verifies that the profiler can correctly profile the network at single active steps.
    Expectation: The profiler should profile the network without any exceptions and
    generate the expected profiling data.
    """
    step_num = 15
    with tempfile.TemporaryDirectory(suffix="_step_profiler_1") as tmpdir:
        schedule = ms.profiler.schedule(wait=1, warmup=1, active=1, repeat=2, skip_first=1)
        add = TinyAddNet()
        _dynamic_step_train_profiler(tmpdir, add, step_num, schedule, ms.GRAPH_MODE, "O0")
        # Check whether the number of generated files is the same as the data collected by the step
        ascend_ms_dir_nums = len([d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))])
        assert ascend_ms_dir_nums == 2
        # Check kernel_details.csv
        kernel_details_path_step_1 = os.path.join(
            tmpdir,
            _sort_directories_by_timestamp(tmpdir)[0],  # The first sorted directory
            "ASCEND_PROFILER_OUTPUT",
            "kernel_details.csv"
        )
        kernel_details_path_step_2 = os.path.join(
            tmpdir,
            _sort_directories_by_timestamp(tmpdir)[1],  # The first sorted directory
            "ASCEND_PROFILER_OUTPUT",
            "kernel_details.csv"
        )
        FileChecker.check_csv_items(kernel_details_path_step_1, {"Step ID": "3"}, fuzzy_match=False)
        FileChecker.check_csv_items(kernel_details_path_step_2, {"Step ID": "6"}, fuzzy_match=False)
        # Check trace_view.json
        trace_view_json_path_1 = os.path.join(
            tmpdir,
            _sort_directories_by_timestamp(tmpdir)[0],  # The first sorted directory
            "ASCEND_PROFILER_OUTPUT",
            "trace_view.json"
        )
        trace_view_json_path_2 = os.path.join(
            tmpdir,
            _sort_directories_by_timestamp(tmpdir)[1],  # The first sorted directory
            "ASCEND_PROFILER_OUTPUT",
            "trace_view.json"
        )
        FileChecker.check_timeline_values(
            trace_view_json_path_1,
            "name",
            [
                "*ProfilerStep#3"  # check profiler step
            ],
            fuzzy_match=True
        )
        FileChecker.check_timeline_values(
            trace_view_json_path_2,
            "name",
            [
                "*ProfilerStep#6"  # check profiler step
            ],
            fuzzy_match=True
        )


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dynamic_step_multi_active_kbk_profiler():
    """
    Feature: Dynamic Step Profiler
    Description: This test case verifies that the profiler can correctly profile the network at multi active steps.
    Expectation: The profiler should profile the network without any exceptions and
    generate the expected profiling data.
    """
    step_num = 10
    with tempfile.TemporaryDirectory(suffix="_step_profiler_2") as tmpdir:
        schedule = ms.profiler.schedule(wait=1, warmup=1, active=2, repeat=2, skip_first=1)
        add = TinyAddNet()
        _dynamic_step_train_profiler(tmpdir, add, step_num, schedule, ms.GRAPH_MODE, "O0")
        # Check whether the number of generated files is the same as the data collected by the step
        ascend_ms_dir_nums = len([d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))])
        assert ascend_ms_dir_nums == 2
        # Check whether the kernel.csv contains the step id column
        kernel_details_path_step_1 = os.path.join(
            tmpdir,
            _sort_directories_by_timestamp(tmpdir)[0],  # The first sorted directory
            "ASCEND_PROFILER_OUTPUT",
            "kernel_details.csv"
        )
        kernel_details_path_step_2 = os.path.join(
            tmpdir,
            _sort_directories_by_timestamp(tmpdir)[1],  # The first sorted directory
            "ASCEND_PROFILER_OUTPUT",
            "kernel_details.csv"
        )
        FileChecker.check_csv_items(kernel_details_path_step_1, {"Step ID": ["3", "4"]}, fuzzy_match=False)
        FileChecker.check_csv_items(kernel_details_path_step_2, {"Step ID": ["7", "8"]}, fuzzy_match=False)
        trace_view_json_path_1 = os.path.join(
            tmpdir,
            _sort_directories_by_timestamp(tmpdir)[0],  # The first sorted directory
            "ASCEND_PROFILER_OUTPUT",
            "trace_view.json"
        )
        trace_view_json_path_2 = os.path.join(
            tmpdir,
            _sort_directories_by_timestamp(tmpdir)[1],  # The first sorted directory
            "ASCEND_PROFILER_OUTPUT",
            "trace_view.json"
        )
        # Check trace_view.json
        FileChecker.check_timeline_values(
            trace_view_json_path_1,
            "name",
            [
                "*ProfilerStep#3",  # check profiler step
                "*ProfilerStep#4",  # check profiler step
            ],
            fuzzy_match=True
        )
        FileChecker.check_timeline_values(
            trace_view_json_path_2,
            "name",
            [
                "*ProfilerStep#7",  # check profiler step
                "*ProfilerStep#8",  # check profiler step
            ],
            fuzzy_match=True
        )


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_step_single_active_py_native_profiler():
    """
    Feature: Dynamic Step Profiler
    Description: This test case verifies that the profiler can correctly profile the network at single active steps.
    Expectation: The profiler should profile the network without any exceptions and
    generate the expected profiling data.
    """
    step_num = 8
    with tempfile.TemporaryDirectory(suffix="_step_profiler_1") as tmpdir:
        schedule = ms.profiler.schedule(wait=1, warmup=1, active=1, repeat=2, skip_first=1)
        net = Net()
        context.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")
        profiler = Profiler(profiler_level=ProfilerLevel.Level2,
                            data_process=False,
                            l2_cache=True,
                            output_path=tmpdir,
                            schedule=schedule,
                            on_trace_ready=prof.tensor_board_trace_handler)
        for _ in range(step_num):
            train_net(net)
            profiler.step()
        # Check whether the number of generated files is the same as the data collected by the step
        ascend_ms_dir_nums = len([d for d in os.listdir(tmpdir) if os.path.isdir(os.path.join(tmpdir, d))])
        assert ascend_ms_dir_nums == 2
        # Check whether the kernel.csv contains the step id column
        kernel_details_path_step_1 = os.path.join(
            tmpdir,
            _sort_directories_by_timestamp(tmpdir)[0],  # The first sorted directory
            "ASCEND_PROFILER_OUTPUT",
            "kernel_details.csv"
        )
        kernel_details_path_step_2 = os.path.join(
            tmpdir,
            _sort_directories_by_timestamp(tmpdir)[1],  # The first sorted directory
            "ASCEND_PROFILER_OUTPUT",
            "kernel_details.csv"
        )
        # Check whether the kernel.csv contains the step id column(to prevent empty steps)
        df1 = pd.read_csv(kernel_details_path_step_1)["Step ID"].tolist()
        df2 = pd.read_csv(kernel_details_path_step_2)["Step ID"].tolist()
        assert all(step_id == 3 for step_id in df1)
        assert all(step_id == 6 for step_id in df2)
        # Check trace_view.json
        trace_view_json_path_1 = os.path.join(
            tmpdir,
            _sort_directories_by_timestamp(tmpdir)[0],  # The first sorted directory
            "ASCEND_PROFILER_OUTPUT",
            "trace_view.json"
        )
        trace_view_json_path_2 = os.path.join(
            tmpdir,
            _sort_directories_by_timestamp(tmpdir)[1],  # The first sorted directory
            "ASCEND_PROFILER_OUTPUT",
            "trace_view.json"
        )
        FileChecker.check_timeline_values(
            trace_view_json_path_1,
            "name",
            [
                "*ProfilerStep#3"  # check profiler step
            ],
            fuzzy_match=True
        )
        FileChecker.check_timeline_values(
            trace_view_json_path_2,
            "name",
            [
                "*ProfilerStep#6"  # check profiler step
            ],
            fuzzy_match=True
        )


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_step_npu_py_native_profiler():
    """
    Feature: Dynamic Step Profiler
    Description: This test case verifies that the profiler can correctly profile the network at CPU device.
    Expectation: The profiler should profile the network without any exceptions and
    generate the expected profiling data.
    """
    step_num = 8
    with tempfile.TemporaryDirectory(suffix="_step_profiler_cpu") as tmpdir:
        schedule = ms.profiler.schedule(wait=1, warmup=1, active=1, repeat=1, skip_first=1)
        net = Net()
        context.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")
        profiler = Profiler(activities=[ProfilerActivity.NPU],
                            output_path=tmpdir,
                            schedule=schedule,
                            on_trace_ready=prof.tensor_board_trace_handler)
        for _ in range(step_num):
            train_net(net)
            profiler.step()
        # Check kernel_details.csv
        kernel_details_path = glob.glob(f"{tmpdir}/*_ascend_ms/"
                                        f"ASCEND_PROFILER_OUTPUT/kernel_details.csv")[0]
        FileChecker.assert_csv_no_header(kernel_details_path, "Step ID")
        FileChecker.check_csv_items(kernel_details_path, {"Name": ["*BiasAdd*", "*MatMul*"]})
        # Check trace_view.json
        trace_view_path = glob.glob(f"{tmpdir}/*_ascend_ms/"
                                    f"ASCEND_PROFILER_OUTPUT/trace_view.json")[0]
        FileChecker.check_timeline_values(
            trace_view_path,
            "name",
            ["*MatMul*",
             "*Add*"
             ],
            fuzzy_match=True
        )


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_step_cpu_py_native_profiler():
    """
    Feature: Dynamic Step Profiler
    Description: This test case verifies that the profiler can correctly profile the network at CPU device.
    Expectation: The profiler should profile the network without any exceptions and
    generate the expected profiling data.
    """
    step_num = 8
    with tempfile.TemporaryDirectory(suffix="_step_profiler_cpu") as tmpdir:
        schedule = ms.profiler.schedule(wait=1, warmup=1, active=1, repeat=1, skip_first=1)
        net = Net()
        context.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")
        profiler = Profiler(activities=[ProfilerActivity.CPU],
                            output_path=tmpdir,
                            schedule=schedule,
                            on_trace_ready=prof.tensor_board_trace_handler)
        for _ in range(step_num):
            train_net(net)
            profiler.step()
        # Check kernel_details.csv
        kernel_details_files = glob.glob(f"{tmpdir}/*_ascend_ms/"
                                         f"ASCEND_PROFILER_OUTPUT/kernel_details.csv")
        assert not kernel_details_files, "kernel_details.csv is exit."
        # Check trace_view.json
        trace_view_path = glob.glob(f"{tmpdir}/*_ascend_ms/"
                                    f"ASCEND_PROFILER_OUTPUT/trace_view.json")[0]
        FileChecker.check_timeline_values(
            trace_view_path,
            "name",
            ["*ProfilerStep#3"
             ],
            fuzzy_match=True
        )


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_step_npu_graph_profiler():
    """
    Feature: Dynamic Step Profiler
    Description: This test case verifies that the profiler can correctly profile the network at CPU device.
    Expectation: The profiler should profile the network without any exceptions and
    generate the expected profiling data.
    """
    step_num = 8
    with tempfile.TemporaryDirectory(suffix="_step_profiler_cpu") as tmpdir:
        schedule = ms.profiler.schedule(wait=1, warmup=1, active=1, repeat=1, skip_first=1)
        net = Net()
        context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")
        context.set_context(jit_config={"jit_level": "O0"})
        profiler = Profiler(activities=[ProfilerActivity.NPU],
                            output_path=tmpdir,
                            schedule=schedule,
                            on_trace_ready=prof.tensor_board_trace_handler)
        for _ in range(step_num):
            train_net(net)
            profiler.step()
        # Check kernel_details.csv
        kernel_details_path = glob.glob(f"{tmpdir}/*_ascend_ms/"
                                        f"ASCEND_PROFILER_OUTPUT/kernel_details.csv")[0]
        FileChecker.assert_csv_no_header(kernel_details_path, "Step ID")
        FileChecker.check_csv_items(kernel_details_path, {"Name": ["*BiasAdd*", "*MatMul*"]})
        # Check trace_view.json
        trace_view_path = glob.glob(f"{tmpdir}/*_ascend_ms/"
                                    f"ASCEND_PROFILER_OUTPUT/trace_view.json")[0]
        FileChecker.check_timeline_values(
            trace_view_path,
            "name",
            ["*MatMul*",
             "*Add*"
             ],
            fuzzy_match=True
        )


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_step_cpu_graph_profiler():
    """
    Feature: Dynamic Step Profiler
    Description: This test case verifies that the profiler can correctly profile the network at CPU device.
    Expectation: The profiler should profile the network without any exceptions and
    generate the expected profiling data.
    """
    step_num = 8
    with tempfile.TemporaryDirectory(suffix="_step_profiler_cpu") as tmpdir:
        schedule = ms.profiler.schedule(wait=1, warmup=1, active=1, repeat=1, skip_first=1)
        net = Net()
        context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")
        context.set_context(jit_config={"jit_level": "O0"})
        profiler = Profiler(activities=[ProfilerActivity.CPU],
                            output_path=tmpdir,
                            schedule=schedule,
                            on_trace_ready=prof.tensor_board_trace_handler)
        for _ in range(step_num):
            train_net(net)
            profiler.step()
        # Check kernel_details.csv
        kernel_details_files = glob.glob(f"{tmpdir}/*_ascend_ms/"
                                         f"ASCEND_PROFILER_OUTPUT/kernel_details.csv")
        assert not kernel_details_files, "kernel_details.csv is exit."
        # Check trace_view.json
        trace_view_path = glob.glob(f"{tmpdir}/*_ascend_ms/"
                                    f"ASCEND_PROFILER_OUTPUT/trace_view.json")[0]
        FileChecker.check_timeline_values(
            trace_view_path,
            "name",
            ["*ProfilerStep#3"
             ],
            fuzzy_match=True
        )


def _dynamic_step_train_profiler(tmpdir, net, step_num, schedule, context_mode, jit_level=None):
    """ Collect performance data according to step"""
    context.set_context(mode=context_mode, device_target="Ascend")
    if jit_level:
        context.set_context(jit_config={"jit_level": jit_level})
    with Profiler(profiler_level=ProfilerLevel.Level0,
                  data_process=False,
                  l2_cache=True,
                  output_path=tmpdir,
                  schedule=schedule,
                  on_trace_ready=prof.tensor_board_trace_handler) as profiler:
        for _ in range(step_num):
            train(net)
            profiler.step()
    ProfilerInterface.finalize()
    ProfilerInterface.clear()


def _extract_timestamp(folder_name):
    """ Extracts a timestamp from a folder name using regular expressions. """
    # Use regular expressions to extract timestamps from folder names
    match = folder_name.split("_")[-3]
    if match:
        return int(match)
    return None


def _sort_directories_by_timestamp(path):
    """ Sorts the first-level directories in a given path based on timestamps in their names. """
    # Check if the path exists
    if not os.path.exists(path):
        print("The path does not exist")
        return []

    # Get all the first-level folders, and extract the timestamps
    directories_with_timestamp = []
    for name in os.listdir(path):
        if os.path.isdir(os.path.join(path, name)):
            timestamp = _extract_timestamp(name)
            if timestamp:
                directories_with_timestamp.append((name, timestamp))

    # Sort folders based on timestamps
    directories_with_timestamp.sort(key=lambda x: x[1])

    # Returns the sorted list of folder names
    return [name for name, _ in directories_with_timestamp]
