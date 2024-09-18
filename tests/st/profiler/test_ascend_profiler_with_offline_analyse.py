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
import shutil
import mindspore
from mindspore import context
from mindspore import Profiler
from mindspore import Tensor

from tests.mark_utils import arg_mark
from file_check import FileChecker
from model_zoo import TinyAddNet


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ascend_profiler_offline_analyse_with_single_device():
    """
    Feature: Ascend Offline Profiler Analysis with Single Device
    Description: Execute offline analysis with the Ascend profiler in PyNative mode for a single device, using a simple
                 addition network.
    Expectation: The profiler successfully performs offline analysis, and the expected analysis files are generated in
                 the temporary directory.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    with tempfile.TemporaryDirectory() as tmpdir:
        rank_id = int(os.getenv('RANK_ID')) if os.getenv('RANK_ID') else 0
        profiler = Profiler(output_path=tmpdir)
        net = TinyAddNet()
        t0 = Tensor(dtype=mindspore.float32, shape=[32, None])
        t1 = Tensor(dtype=mindspore.float32, shape=[32, None])
        net(t0, t1)
        profiler.stop()
        profiler._ascend_profiler.finalize()
        profiler.offline_analyse(path=tmpdir, data_simplification=False)
        check_ascend_offline_analyse_files(tmpdir, rank_id)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ascend_profiler_offline_analyse_with_multi_devices():
    """
    Feature: Ascend Offline Profiler Analysis with Multi Devices
    Description: Test the Ascend profiler offline analysis capability in PyNative mode with multiple devices, using
                 a simple addition network.
    Expectation: Profiler offline analysis should successfully process data from multiple devices, with results
                 properly generated in the corresponding rank directories.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    with tempfile.TemporaryDirectory() as tmpdir:
        rank_id = int(os.getenv('RANK_ID')) if os.getenv('RANK_ID') else 0
        profiler = Profiler(output_path=tmpdir)
        net = TinyAddNet()
        t0 = Tensor(dtype=mindspore.float32, shape=[32, None])
        t1 = Tensor(dtype=mindspore.float32, shape=[32, None])
        net(t0, t1)
        profiler.stop()
        profiler._ascend_profiler.finalize()
        # copy profiler data to rank0 and rank1
        rank0_path = os.path.join(tmpdir, 'rank0')
        rank1_path = os.path.join(tmpdir, 'rank1')
        os.makedirs(rank0_path, exist_ok=True)
        os.makedirs(rank1_path, exist_ok=True)
        profiler_dir = os.path.join(tmpdir, 'profiler')
        shutil.copytree(profiler_dir, os.path.join(rank0_path, 'profiler'))
        shutil.copytree(profiler_dir, os.path.join(rank1_path, 'profiler'))
        shutil.rmtree(profiler_dir)  # remove the original profiler data.
        profiler.offline_analyse(path=tmpdir, data_simplification=False)
        check_ascend_offline_analyse_files(rank0_path, rank_id)
        check_ascend_offline_analyse_files(rank1_path, rank_id)


def check_ascend_offline_analyse_files(profiler_path: str, rank_id: int):
    ascend_profiler_output_path = glob.glob(f"{profiler_path}/profiler/rank-*_ascend_ms/ASCEND_PROFILER_OUTPUT")[0]

    # Check trace_view.json
    trace_view_path = os.path.join(ascend_profiler_output_path, "trace_view.json")
    FileChecker.check_timeline_values(
        trace_view_path,
        "name",
        [
            "AscendCL@*",     # check CANN trace
            "HostToDevice*",  # check HostToDevice flow
        ],
        fuzzy_match=True
    )

    # check op_statistic.csv
    op_statistic_path = os.path.join(ascend_profiler_output_path, "op_statistic.csv")
    FileChecker.check_csv_items(op_statistic_path, {"OP Type": ["*Add*"]})

    # check aicore_intermediate_*_type.csv
    aicore_intermediate_type_path = os.path.join(profiler_path, "profiler", f"aicore_intermediate_{rank_id}_type.csv")
    FileChecker.check_csv_items(aicore_intermediate_type_path, {"kernel_type": ["*Add*"]})

    # check aicore_intermediate_*_detail.csv
    aicore_intermediate_detail_path = os.path.join(profiler_path, "profiler",
                                                   f"aicore_intermediate_{rank_id}_detail.csv")
    FileChecker.check_csv_items(aicore_intermediate_detail_path, {"full_kernel_name": "*Add*"})
