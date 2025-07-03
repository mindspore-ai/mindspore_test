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
"""PanguAlpha model"""

import os
import mindspore.common.dtype as mstype
from mindspore.nn.wrap.cell_wrapper import _VirtualDatasetCell
from mindspore.nn import PipelineCell
from mindspore import context, Tensor
from mindspore import Symbol
from tests.ut.python.parallel.test_dynamic_pangu_alpha import PanguAlpha, PANGUALPHAConfig, CrossEntropyLoss, PanguAlphaWithLossLazyInline, compile_pipeline_net


context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
def test_pipeline_dp_mp_op_bs_and_seq_dynamic_cell_reuse_stage0():
    '''
    Feature: batch dim and seq dim are dynamic, and using pp + dp + mp + op, cell_reuse, test stage-0
    Description: all reshape skip redistribution, pipeline slice micro skip redistribution, and set virtual dataset
    Expectation: compile success
    '''
    context.reset_auto_parallel_context()
    os.environ['PIPELINE_SLICE_SKIP_REDISTRIBUTION'] = "1"
    ds_strategy = ((2, 1), (2, 1), (2, 1, 1))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, dataset_strategy=ds_strategy,
                                      enable_parallel_optimizer=True, global_rank=0,
                                      pipeline_config={"pipeline_scheduler": "1f1b", "pipeline_interleave": True})
    config = PANGUALPHAConfig(data_parallel_num=2, model_parallel_num=2, pipeline_parallel_num=2, num_layers=4)
    pangu_alpha = PanguAlpha(config)
    for i in range(pangu_alpha.backbone.num_layers):
        if i % 2 == 0:
            pangu_alpha.backbone.blocks[i].pipeline_stage = 0
        else:
            pangu_alpha.backbone.blocks[i].pipeline_stage = 1
    loss = CrossEntropyLoss(config)
    loss.pipeline_stage = 1
    pangu_alpha_loss = PanguAlphaWithLossLazyInline(config, pangu_alpha, loss)
    pangu_alpha_loss = PipelineCell(pangu_alpha_loss, 2)
    for i in range(pangu_alpha_loss.micro_size):
        pangu_alpha_loss.micro_inputs[i].strided_slice.add_prim_attr("out_shard_size", 2)
    net = _VirtualDatasetCell(pangu_alpha_loss)
    net._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")

    s1 = Symbol(divisor=8, unique=True)
    s2 = Symbol(divisor=16, remainder=1)
    s3 = Symbol(divisor=16)
    input_ids = Tensor(shape=[s1, s2], dtype=mstype.int32)
    input_position = Tensor(shape=[s1, s3], dtype=mstype.int32)
    attention_mask = Tensor(shape=[s1, s3, s3], dtype=mstype.float16)

    compile_pipeline_net(net, input_ids, input_position, attention_mask)
    del os.environ['PIPELINE_SLICE_SKIP_REDISTRIBUTION']
    context.reset_auto_parallel_context()


def test_pipeline_dp_mp_op_bs_and_seq_dynamic_cell_reuse_stage1():
    '''
    Feature: batch dim and seq dim are dynamic, and using pp + dp + mp + op, cell_reuse, test stage-1
    Description: all reshape skip redistribution, pipeline slice micro skip redistribution, and set virtual dataset
    Expectation: compile success
    '''
    context.reset_auto_parallel_context()
    os.environ['PIPELINE_SLICE_SKIP_REDISTRIBUTION'] = "1"
    ds_strategy = ((2, 1), (2, 1), (2, 1, 1))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, dataset_strategy=ds_strategy,
                                      enable_parallel_optimizer=True, global_rank=4,
                                      pipeline_config={"pipeline_scheduler": "1f1b", "pipeline_interleave": True})
    config = PANGUALPHAConfig(data_parallel_num=2, model_parallel_num=2, pipeline_parallel_num=2, num_layers=3)
    pangu_alpha = PanguAlpha(config)
    for i in range(pangu_alpha.backbone.num_layers):
        if i % 2 == 0:
            pangu_alpha.backbone.blocks[i].pipeline_stage = 0
        else:
            pangu_alpha.backbone.blocks[i].pipeline_stage = 1
    loss = CrossEntropyLoss(config)
    loss.pipeline_stage = 1
    pangu_alpha_loss = PanguAlphaWithLossLazyInline(config, pangu_alpha, loss)
    pangu_alpha_loss = PipelineCell(pangu_alpha_loss, 2)
    for i in range(pangu_alpha_loss.micro_size):
        pangu_alpha_loss.micro_inputs[i].strided_slice.add_prim_attr("out_shard_size", 2)
    net = _VirtualDatasetCell(pangu_alpha_loss)
    net._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")

    s1 = Symbol(divisor=8, unique=True)
    s2 = Symbol(divisor=16, remainder=1)
    s3 = Symbol(divisor=16)
    input_ids = Tensor(shape=[s1, s2], dtype=mstype.int32)
    input_position = Tensor(shape=[s1, s3], dtype=mstype.int32)
    attention_mask = Tensor(shape=[s1, s3, s3], dtype=mstype.float16)

    compile_pipeline_net(net, input_ids, input_position, attention_mask)
    del os.environ['PIPELINE_SLICE_SKIP_REDISTRIBUTION']
    context.reset_auto_parallel_context()
