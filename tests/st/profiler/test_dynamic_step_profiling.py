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
"""test dynamic step profiling"""
import glob
import json
import os
import tempfile

import numpy as np

import mindspore as ms
import mindspore.dataset as ds
from mindspore import context, nn
from mindspore.profiler.dynamic_profiler import DynamicProfilerMonitor
from tests.mark_utils import arg_mark
from file_check import FileChecker


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Dense(2, 2)

    def construct(self, x):
        return self.fc(x)


def generator_net():
    for _ in range(2):
        yield np.ones([2, 2]).astype(np.float32), np.ones([2]).astype(np.int32)


def train(net):
    optimizer = nn.Momentum(net.trainable_params(), 1, 0.9)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    data = ds.GeneratorDataset(generator_net(), ["data", "label"])
    model = ms.train.Model(net, loss, optimizer)
    model.train(1, data)


def train_net_with_dynamic_profiler(output_path, cfg_path):
    net = Net()
    STEP_NUM = 15
    context.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")
    dp = DynamicProfilerMonitor(cfg_path=cfg_path, output_path=output_path)
    for i in range(STEP_NUM):
        train(net)
        if i == 5:
            change_cfg_json(os.path.join(cfg_path, "profiler_config.json"))
        dp.step()


def change_cfg_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    data['start_step'] = 6
    data['stop_step'] = 7

    with open(json_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_step_profiler():
    """
    Feature: Dynamic Step Profiler Testing
    Description: This test function is designed to verify the functionality of the dynamic step profiler.
    Expectation: The test expects to find specific profiling data in the output files of the profiler.
    """
    data_cfg = {
        "start_step": 2,
        "stop_step": 5,
        "aicore_metrics": 1,
        "profiler_level": -1,
        "profile_framework": 1,
        "analyse_mode": 0,
        "with_stack": True,
        "profile_memory": True,
        "parallel_strategy": True,
        "data_simplification": False,
    }
    with tempfile.TemporaryDirectory(suffix="_step_profiler") as tmpdir:
        cfg_path = os.path.join(tmpdir, "profiler_config.json")
        # set cfg file
        with open(cfg_path, 'w') as f:
            json.dump(data_cfg, f, indent=4)

        rank_id = int(os.getenv('RANK_ID')) if os.getenv('RANK_ID') else 0
        train_net_with_dynamic_profiler(output_path=tmpdir, cfg_path=tmpdir)
        profiler_step_2_5_path = os.path.join(tmpdir, f"rank{rank_id}_start2_stop5")
        profiler_step_6_7_path = os.path.join(tmpdir, f"rank{rank_id}_start6_stop7")

        # Check trace_view.json
        trace_view_step_2_5_path = glob.glob(f"{profiler_step_2_5_path}/*_ascend_ms/"
                                             f"ASCEND_PROFILER_OUTPUT/trace_view.json")[0]
        trace_view_step_6_7_path = glob.glob(f"{profiler_step_6_7_path}/*_ascend_ms/"
                                             f"ASCEND_PROFILER_OUTPUT/trace_view.json")[0]
        FileChecker.check_timeline_values(
            trace_view_step_2_5_path,
            "name",
            ["*ProfilerStep#1",
             "*ProfilerStep#2",
             "*ProfilerStep#3",
             "*ProfilerStep#4",
             "*MatMul*",
             "*Add*"
             ],
            fuzzy_match=True
        )
        FileChecker.check_timeline_values(
            trace_view_step_6_7_path,
            "name",
            ["*ProfilerStep#1",
             "*ProfilerStep#2",
             "*MatMul*",
             "*Add*"],
            fuzzy_match=True
        )
        # Check kernel_details.csv
        kernel_details_step_2_5_path = glob.glob(f"{profiler_step_2_5_path}/*_ascend_ms/"
                                                 f"ASCEND_PROFILER_OUTPUT/kernel_details.csv")[0]
        kernel_details_step_6_7_path = glob.glob(f"{profiler_step_6_7_path}/*_ascend_ms/"
                                                 f"ASCEND_PROFILER_OUTPUT/kernel_details.csv")[0]
        FileChecker.check_csv_items(kernel_details_step_2_5_path, {"Step ID": ["1", "2", "3", "4"]},
                                    fuzzy_match=False
                                    )
        FileChecker.check_csv_items(kernel_details_step_2_5_path, {"Name": ["*BiasAdd*", "*MatMul*"]})
        FileChecker.check_csv_items(kernel_details_step_6_7_path, {"Step ID": ["1", "2"]},
                                    fuzzy_match=False
                                    )
        FileChecker.check_csv_items(kernel_details_step_6_7_path, {"Name": ["*BiasAdd*", "*MatMul"]})
        # Check operate_memory.csv
        operate_memory_2_5_path = glob.glob(f"{profiler_step_2_5_path}/*_ascend_ms/"
                                            f"ASCEND_PROFILER_OUTPUT/operator_memory.csv")[0]
        operate_memory_6_7_path = glob.glob(f"{profiler_step_6_7_path}/*_ascend_ms/"
                                            f"ASCEND_PROFILER_OUTPUT/operator_memory.csv")[0]
        FileChecker.check_csv_items(operate_memory_2_5_path, {"Name": ["Unknown"]})
        FileChecker.check_csv_items(operate_memory_6_7_path, {"Name": ["Unknown"]})
