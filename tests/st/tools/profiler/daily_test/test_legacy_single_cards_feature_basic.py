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
"""legacy single"""

import tempfile
from mindspore import context
from mindspore import Profiler
from mindspore.profiler import ProfilerLevel, ProfilerActivity, AicoreMetrics
import mindspore.profiler as Prof

from tests.mark_utils import arg_mark
from tests.st.tools.profiler.model_zoo import TinyTransformer
from tests.st.tools.profiler.fake_dataset import FakeDataset


def generator_profiler_data(tmpdir):
    """
    Collect profiler data.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O2"})

    class Config:
        """config"""
        def __init__(self, output_path):
            self.profiler_dir = output_path
            self.profiler_level = ProfilerLevel.Level1
            self.activities = [ProfilerActivity.CPU, ProfilerActivity.NPU]
            self.aicore_metrics = AicoreMetrics.ArithmeticUtilization
            self.with_stack = False
            self.profile_memory = False
            self.data_process = True
            self.parallel_strategy = False
            self.start_profile = True
            self.l2_cache = False
            self.hbm_ddr = False
            self.pcie = False
            self.sync_enable = True
            self.data_simplification = True
            self.mstx = False

    cfg = Config(tmpdir)

    # Create Profiler instance with all parameters
    profiler = Profiler(
        output_path=cfg.profiler_dir,
        profiler_level=cfg.profiler_level,
        activities=cfg.activities,
        aicore_metrics=cfg.aicore_metrics,
        with_stack=cfg.with_stack,
        profile_memory=cfg.profile_memory,
        data_process=cfg.data_process,
        parallel_strategy=cfg.parallel_strategy,
        start_profile=cfg.start_profile,
        l2_cache=cfg.l2_cache,
        hbm_ddr=cfg.hbm_ddr,
        pcie=cfg.pcie,
        sync_enable=cfg.sync_enable,
        data_simplification=cfg.data_simplification,
        mstx=cfg.mstx,
        on_trace_ready=Prof.tensorboard_trace_handler()
    )

    net = TinyTransformer(d_model=2, nhead=1, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=4)
    nlp_dataset = FakeDataset.create_fake_nlp_dataset(seq_len=1, batch_size=1, d_model=2, tgt_len=1, num_samples=1)
    for src, tgt in nlp_dataset:
        net(src, tgt)

    profiler.analyse()
    profiler.stop()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_func_profiler_single_card_resnet_sink_false_framework_all_14():
    """
    Feature: Profiler iterator data
    Description: Test the profiler analyse method with pretty_on=False to generate compressed JSON output.
    Expectation: run success.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        generator_profiler_data(tmpdir)
