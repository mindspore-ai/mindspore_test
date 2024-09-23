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
import tempfile
from mindspore import context
from mindspore import Profiler

from tests.mark_utils import arg_mark
from file_check import FileChecker
from model_zoo import TinyTransformer
from fake_dataset import FakeDataset


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ascend_graph_mode_profiler_with_static_shape_all_parameters_on():
    """
    Feature: Ascend Graph Mode Profiler with All Parameters Enabled
    Description: Test the Ascend profiler in graph mode with all profiling parameters turned on, using a static shape
                 model.
    Expectation: The profiler collects and analyzes data successfully, and the output files are correctly generated
                 in the temporary directory.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_level="O2")
    with tempfile.TemporaryDirectory() as tmpdir:
        rank_id = int(os.getenv('RANK_ID')) if os.getenv('RANK_ID') else 0
        profiler = Profiler(
            output_path=tmpdir,
            op_time=True,
            profile_communication=True,
            profile_memory=True,
            parallel_strategy=True,
            start_profile=True,
            aicore_metrics=1,
            l2_cache=True,
            hbm_ddr=True,
            pcie=True,
            sync_enable=True,
            data_process=True,
            profile_framework='all',
            with_stack=False,
            data_simplification=False
        )
        net = TinyTransformer(d_model=2, nhead=1, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=4)
        nlp_dataset = FakeDataset.create_fake_nlp_dataset(seq_len=1, batch_size=1, d_model=2, tgt_len=1, num_samples=1)
        for src, tgt in nlp_dataset:
            net(src, tgt)
        profiler.analyse()
        check_ascend_profiler_graph_files(tmpdir, rank_id)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ascend_pynative_mode_profiler_with_static_shape_all_parameters_on():
    """
    Feature: Ascend pynative Mode Profiler with All Parameters Enabled
    Description: Test the Ascend profiler in pynative mode with all profiling parameters turned on, using a static shape
                 model.
    Expectation: The profiler collects and analyzes data successfully, and the output files are correctly generated
                 in the temporary directory.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    with tempfile.TemporaryDirectory() as tmpdir:
        rank_id = int(os.getenv('RANK_ID')) if os.getenv('RANK_ID') else 0
        profiler = Profiler(
            output_path=tmpdir,
            op_time=True,
            profile_communication=True,
            profile_memory=True,
            parallel_strategy=True,
            start_profile=True,
            aicore_metrics=1,
            l2_cache=True,
            hbm_ddr=True,
            pcie=True,
            sync_enable=True,
            data_process=True,
            profile_framework='all',
            with_stack=False,
            data_simplification=False
        )
        net = TinyTransformer(d_model=2, nhead=1, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=4)
        nlp_dataset = FakeDataset.create_fake_nlp_dataset(seq_len=1, batch_size=1, d_model=2, tgt_len=1, num_samples=1)
        for src, tgt in nlp_dataset:
            net(src, tgt)
        profiler.analyse()
        check_ascend_profiler_pynative_files(tmpdir, rank_id)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ascend_kbk_mode_profiler_with_static_shape_all_parameters_on():
    """
    Feature: Ascend kbk Mode Profiler with All Parameters Enabled
    Description: Test the Ascend profiler in kbk mode with all profiling parameters turned on, using a static shape
                 model.
    Expectation: The profiler collects and analyzes data successfully, and the output files are correctly generated
                 in the temporary directory.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_level="O0")
    with tempfile.TemporaryDirectory() as tmpdir:
        rank_id = int(os.getenv('RANK_ID')) if os.getenv('RANK_ID') else 0
        profiler = Profiler(
            output_path=tmpdir,
            op_time=True,
            profile_communication=True,
            profile_memory=True,
            parallel_strategy=True,
            start_profile=True,
            aicore_metrics=1,
            l2_cache=True,
            hbm_ddr=True,
            pcie=True,
            sync_enable=True,
            data_process=True,
            profile_framework='all',
            with_stack=False,
            data_simplification=False
        )
        net = TinyTransformer(d_model=2, nhead=1, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=4)
        nlp_dataset = FakeDataset.create_fake_nlp_dataset(seq_len=1, batch_size=1, d_model=2, tgt_len=1, num_samples=1)
        for src, tgt in nlp_dataset:
            net(src, tgt)
        profiler.analyse()
        check_ascend_profiler_kbk_files(tmpdir, rank_id)


def check_ascend_profiler_all_parameters_on_common_files(profiler_path: str, rank_id: int):
    ascend_profiler_output_path = glob.glob(f"{profiler_path}/profiler/rank-*_ascend_ms/ASCEND_PROFILER_OUTPUT")[0]
    msprof_path = glob.glob(f"{profiler_path}/profiler/PROF_*/mindstudio_profiler_output")[0]

    # check hbm_*.csv
    hbm_path = glob.glob(f"{msprof_path}/hbm_*")[0]
    FileChecker.check_csv_headers(hbm_path, ["Device_id", "Metric", "Read(MB/s)", "Write(MB/s)"])

    # check l2_cache_*.csv
    l2_cache_path = glob.glob(f"{msprof_path}/l2_cache_*")[0]
    FileChecker.check_csv_items(l2_cache_path, {
        "Op Name": ["*Add*", "*MatMul*", "*LayerNorm*"]
    })

    # check pcie_*.csv
    pcie_path = glob.glob(f"{msprof_path}/pcie_*")[0]
    FileChecker.check_csv_headers(pcie_path, ["Device_id", "Mode", "Min", "Max", "Avg"])

    # Check trace_view.json
    trace_view_path = os.path.join(ascend_profiler_output_path, "trace_view.json")
    FileChecker.check_timeline_values(
        trace_view_path,
        "name",
        [
            "Dataset::*",            # check dataset trace
            "PynativeFramework::*",  # check host trace
            "AscendCL@*",            # check CANN trace
            "GetNext*",              # check kernel on Ascend Hardware
            "Free",                  # check overlay analysis
            "HostToDevice*",         # check HostToDevice flow
        ],
        fuzzy_match=True
    )

    # check op_statistic.csv
    op_statistic_path = os.path.join(ascend_profiler_output_path, "op_statistic.csv")
    FileChecker.check_csv_items(op_statistic_path, {
        "OP Type": ["*Add*", "*MatMul*", "*LayerNorm*"]
    })

    # check kernel_details.csv
    kernel_details_path = os.path.join(ascend_profiler_output_path, "kernel_details.csv")
    FileChecker.check_csv_items(kernel_details_path, {
        "Name": ["*Add*", "*MatMul*", "*LayerNorm*"]
    })

    # check profile_info_*.json
    profile_info_path = os.path.join(profiler_path, "profiler", f"profiler_info_{rank_id}.json")
    FileChecker.check_json_items(profile_info_path, {
        "profiling_options.aicpu": "on",
        "profiling_options.op_time": "on",
        "profiling_options.profile_memory": "on",
        "data_simplification": False
    })

    # check dataset_*.csv
    dataset_path = os.path.join(profiler_path, "profiler", f"dataset_{rank_id}.csv")
    FileChecker.check_csv_items(dataset_path, {
        "Operation": ["Pipeline", "RandomDataOp"]
    })

    # check minddata_pipeline_summary_0.csv
    minddata_pipeline_summary_path = os.path.join(profiler_path, "profiler", f"minddata_pipeline_summary_{rank_id}.csv")
    FileChecker.check_csv_items(minddata_pipeline_summary_path, {
        "pipeline_ops": ["op_names", "op_ids", "num_workers"]
    })


def check_ascend_profiler_graph_files(profiler_path: str, rank_id: int):
    check_ascend_profiler_all_parameters_on_common_files(profiler_path, rank_id)
    ascend_profiler_output_path = glob.glob(f"{profiler_path}/profiler/rank-*_ascend_ms/ASCEND_PROFILER_OUTPUT")[0]

    # check operate_memory.csv
    operate_memory_path = os.path.join(ascend_profiler_output_path, "operator_memory.csv")
    FileChecker.check_csv_items(operate_memory_path, {
        "Name": ["model.encoder*", "model.decoder*"]
    })

    # check static_op_mem.csv
    static_op_mem_path = os.path.join(ascend_profiler_output_path, "static_op_mem.csv")
    FileChecker.check_csv_items(static_op_mem_path, {
        "Op Name": ["*Add*", "*LayerNorm*"]
    })

    # check step_trace_raw_*_detail_time.csv
    step_trace_raw_detail_time_path = os.path.join(profiler_path, "profiler",
                                                   f"step_trace_raw_{rank_id}_detail_time.csv")
    FileChecker.check_file_line_count(step_trace_raw_detail_time_path, 3)


def check_ascend_profiler_pynative_files(profiler_path: str, rank_id: int):
    check_ascend_profiler_all_parameters_on_common_files(profiler_path, rank_id)
    ascend_profiler_output_path = glob.glob(f"{profiler_path}/profiler/rank-*_ascend_ms/ASCEND_PROFILER_OUTPUT")[0]

    # check operate_memory.csv
    operate_memory_path = os.path.join(ascend_profiler_output_path, "operator_memory.csv")
    FileChecker.check_csv_items(operate_memory_path, {
        "Name": ["*Add*", "*Sqrt*", "*LayerNorm*"]
    })


def check_ascend_profiler_kbk_files(profiler_path: str, rank_id: int):
    check_ascend_profiler_all_parameters_on_common_files(profiler_path, rank_id)
    ascend_profiler_output_path = glob.glob(f"{profiler_path}/profiler/rank-*_ascend_ms/ASCEND_PROFILER_OUTPUT")[0]

    # check operate_memory.csv
    operate_memory_path = os.path.join(ascend_profiler_output_path, "operator_memory.csv")
    FileChecker.check_csv_items(operate_memory_path, {
        "Name": ["*Add*", "*MatMul*", "*LayerNorm*"]
    })
