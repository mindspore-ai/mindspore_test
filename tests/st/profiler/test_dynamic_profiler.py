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
import glob
import json
import tempfile
from tests.mark_utils import arg_mark
from model_zoo import TinyTransformer
from fake_dataset import FakeDataset
from file_check import FileChecker
import mindspore as ms
from mindspore import context
from mindspore.train import Model
from mindspore.profiler import DynamicProfilerMonitor


class StepMonitor(ms.Callback):
    def on_train_step_begin(self, run_context):
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        print(f"-------------- Step {step_num} begin ----------------")

    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        print(f"-------------- Step {step_num} end ----------------")



def train_tiny_transformer_with_dynamic_profiler(output_path, cfg_path):
    ds_train = FakeDataset.create_fake_nlp_dataset(
        seq_len=1,
        batch_size=1,
        d_model=2,
        tgt_len=1,
        num_samples=5,
        num_parallel_workers=1
    )

    network = TinyTransformer(
        d_model=2,
        nhead=1,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=2
    )

    profile_callback = DynamicProfilerMonitor(cfg_path=cfg_path, output_path=output_path)
    step_cb = StepMonitor()
    model = Model(network)
    model.train(1, ds_train, callbacks=[profile_callback, step_cb], dataset_sink_mode=False)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tiny_transformer_pynative_with_dynamic_profiler():
    """
    Feature: DynamicProfilerMonitor
    Description: Test the Ascend profiler in pynative mode with DynamicProfilerMonitor, using a static shape
                 model tiny transformer.
    Expectation: The profiler collects and analyzes data successfully, and the output files are correctly generated
                 in the temporary directory.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    data_cfg = {
        "start_step": 2,
        "stop_step": 3,
        "aicore_metrics": 1,
        "profiler_level": -1,
        "profile_framework": 1,
        "analyse_mode": 0,
        "with_stack": True,
        "parallel_strategy": True,
        "data_simplification": False,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = os.path.join(tmpdir, "profiler_config.json")
        # set cfg file
        with open(cfg_path, 'w') as f:
            json.dump(data_cfg, f, indent=4)

        rank_id = int(os.getenv('RANK_ID')) if os.getenv('RANK_ID') else 0
        train_tiny_transformer_with_dynamic_profiler(output_path=tmpdir, cfg_path=tmpdir)
        profiler_path = os.path.join(tmpdir, f"rank{rank_id}_start2_stop3")

        # Check trace_view.json
        trace_view_path = glob.glob(f"{profiler_path}/profiler/rank-*_ascend_ms/"
                                    f"ASCEND_PROFILER_OUTPUT/trace_view.json")[0]
        FileChecker.check_timeline_values(
            trace_view_path,
            "name",
            [
                "mindspore/nn/*",  # check stack trace
                "PynativeFramework::*",  # check host trace
                "AscendCL@*",  # check CANN trace
                "aclnn*",  # check kernel on Ascend Hardware
                "Free",  # check overlay analysis
                "HostToDevice*",  # check HostToDevice flow
                "mindspore_to_npu"  # check mindspore_to_npu
            ],
            fuzzy_match=True
        )

        # check op_statistic.csv
        op_statistic_path = glob.glob(f"{profiler_path}/profiler/rank-*_ascend_ms/"
                                      f"ASCEND_PROFILER_OUTPUT/op_statistic.csv")[0]
        FileChecker.check_csv_items(op_statistic_path, {"OP Type": ["MatMul*", "LayerNorm*"]})

        # check kernel_details.csv
        kernel_details_path = glob.glob(f"{profiler_path}/profiler/rank-*_ascend_ms/"
                                        f"ASCEND_PROFILER_OUTPUT/kernel_details.csv")[0]
        FileChecker.check_csv_items(kernel_details_path, {"Name": ["*MatMul*", "LayerNorm*"]})

        # check step_trace_time.csv
        step_trace_time_path = glob.glob(f"{profiler_path}/profiler/rank-*_ascend_ms/"
                                         f"ASCEND_PROFILER_OUTPUT/step_trace_time.csv")[0]
        FileChecker.check_file_line_count(step_trace_time_path, 2)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_tiny_transformer_kbk_with_dynamic_profiler():
    """
    Feature: DynamicProfilerMonitor
    Description: Test the Ascend profiler in KBK mode with DynamicProfilerMonitor, using a static shape
                 model tiny transformer.
    Expectation: The profiler collects and analyzes data successfully, and the output files are correctly generated
                 in the temporary directory.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_level="O0")
    data_cfg = {
        "start_step": 2,
        "stop_step": 3,
        "aicore_metrics": 1,
        "profiler_level": -1,
        "profile_framework": 1,
        "analyse_mode": 0,
        "with_stack": True,
        "parallel_strategy": True,
        "data_simplification": False,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = os.path.join(tmpdir, "profiler_config.json")
        # set cfg file
        with open(cfg_path, 'w') as f:
            json.dump(data_cfg, f, indent=4)

        rank_id = int(os.getenv('RANK_ID')) if os.getenv('RANK_ID') else 0
        train_tiny_transformer_with_dynamic_profiler(output_path=tmpdir, cfg_path=tmpdir)
        profiler_path = os.path.join(tmpdir, f"rank{rank_id}_start2_stop3")

        # Check trace_view.json
        trace_view_path = glob.glob(f"{profiler_path}/profiler/rank-*_ascend_ms/"
                                    f"ASCEND_PROFILER_OUTPUT/trace_view.json")[0]
        FileChecker.check_timeline_values(
            trace_view_path,
            "name",
            [
                "mindspore/nn/*",  # check stack trace
                "Kernel::*",  # check host trace
                "AscendCL@*",  # check CANN trace
                "model-Transformer", # check scope layer
                "aclnn*",  # check kernel on Ascend Hardware
                "Free",  # check overlay analysis
                "HostToDevice*",  # check HostToDevice flow
                "mindspore_to_npu"  # check mindspore_to_npu
            ],
            fuzzy_match=True
        )

        # check op_statistic.csv
        op_statistic_path = glob.glob(f"{profiler_path}/profiler/rank-*_ascend_ms/"
                                      f"ASCEND_PROFILER_OUTPUT/op_statistic.csv")[0]
        FileChecker.check_csv_items(op_statistic_path, {"OP Type": ["MatMul*", "LayerNorm*"]})

        # check kernel_details.csv
        kernel_details_path = glob.glob(f"{profiler_path}/profiler/rank-*_ascend_ms/"
                                        f"ASCEND_PROFILER_OUTPUT/kernel_details.csv")[0]
        FileChecker.check_csv_items(kernel_details_path, {"Name": ["*MatMul*", "LayerNorm*"]})

        # check step_trace_time.csv
        step_trace_time_path = glob.glob(f"{profiler_path}/profiler/rank-*_ascend_ms/"
                                         f"ASCEND_PROFILER_OUTPUT/step_trace_time.csv")[0]
        FileChecker.check_file_line_count(step_trace_time_path, 2)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_tiny_transformer_o2_with_dynamic_profiler():
    """
    Feature: DynamicProfilerMonitor
    Description: Test the Ascend profiler in GRAPH mode with DynamicProfilerMonitor, using a static shape
                 model tiny transformer.
    Expectation: The profiler collects and analyzes data successfully, and the output files are correctly generated
                 in the temporary directory.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_level="O2")
    data_cfg = {
        "start_step": 2,
        "stop_step": 3,
        "aicore_metrics": 1,
        "profiler_level": -1,
        "profile_framework": 1,
        "analyse_mode": 0,
        "with_stack": True,
        "parallel_strategy": True,
        "data_simplification": False,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = os.path.join(tmpdir, "profiler_config.json")
        # set cfg file
        with open(cfg_path, 'w') as f:
            json.dump(data_cfg, f, indent=4)

        rank_id = int(os.getenv('RANK_ID')) if os.getenv('RANK_ID') else 0
        train_tiny_transformer_with_dynamic_profiler(output_path=tmpdir, cfg_path=tmpdir)
        profiler_path = os.path.join(tmpdir, f"rank{rank_id}_start2_stop3")

        # Check trace_view.json
        trace_view_path = glob.glob(f"{profiler_path}/profiler/rank-*_ascend_ms/"
                                    f"ASCEND_PROFILER_OUTPUT/trace_view.json")[0]
        FileChecker.check_timeline_values(
            trace_view_path,
            "name",
            [
                "mindspore/nn/*",  # check stack trace
                "Kernel::*",  # check host trace
                "Model@ModelLoad",  # check CANN trace
                "model-Transformer", # check scope layer
                "*MatMul*",  # check kernel on Ascend Hardware
                "Free",  # check overlay analysis
                "HostToDevice*",  # check HostToDevice flow
            ],
            fuzzy_match=True
        )

        # check op_statistic.csv
        op_statistic_path = glob.glob(f"{profiler_path}/profiler/rank-*_ascend_ms/"
                                      f"ASCEND_PROFILER_OUTPUT/op_statistic.csv")[0]
        FileChecker.check_csv_items(op_statistic_path, {"OP Type": ["MatMul*", "LayerNorm*"]})

        # check kernel_details.csv
        kernel_details_path = glob.glob(f"{profiler_path}/profiler/rank-*_ascend_ms/"
                                        f"ASCEND_PROFILER_OUTPUT/kernel_details.csv")[0]
        FileChecker.check_csv_items(kernel_details_path, {"Name": ["*MatMul*", "LayerNorm*"]})

        # check step_trace_time.csv
        step_trace_time_path = glob.glob(f"{profiler_path}/profiler/rank-*_ascend_ms/"
                                         f"ASCEND_PROFILER_OUTPUT/step_trace_time.csv")[0]
        FileChecker.check_file_line_count(step_trace_time_path, 2)
