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
import os
import numpy as np
import mindspore as ms
from mindspore.parallel import rank_list_for_convert, convert_checkpoint_by_rank, convert_checkpoints
import mindspore.common.dtype as mstype
from mindspore import context
from mindspore.communication.management import init, get_rank
from mindspore.parallel.shard import Layout
from .checkpoints_transform import TestNet, compile_net_and_save_ckpt, compare_ckpt_and_network_params, clean_ckpts


def run_convert(src_strategy_file, dst_strategy_file, src_ckpt_path, dst_ckpt_path, use_safetensor=False):
    target_rank = get_rank()
    if use_safetensor:
        if get_rank() == 0:
            convert_checkpoints(src_ckpt_path, dst_ckpt_path, 'checkpoint_',
                                src_strategy_file, dst_strategy_file, output_format='safetensors')
    else:
        rank_list = rank_list_for_convert(
            target_rank, src_strategy_file, dst_strategy_file)
        checkpoint_files_map = {}
        for rank in rank_list:
            checkpoint_files_map[rank] = "{path}/rank_{rank}/checkpoint_{rank}.ckpt".format(
                path=src_ckpt_path, rank=rank)
        save_checkpoint_file_name = os.path.join(dst_ckpt_path, "rank_{rank}/checkpoint_{rank}.ckpt".
                                                 format(rank=target_rank))
        if not os.path.exists(os.path.join(dst_ckpt_path, 'rank_{}'.format(get_rank()))):
            os.makedirs(os.path.join(dst_ckpt_path,
                                     'rank_{}'.format(get_rank())), exist_ok=True)
        convert_checkpoint_by_rank(target_rank, checkpoint_files_map, save_checkpoint_file_name,
                                   src_strategy_file, dst_strategy_file)
    ms.mint.distributed.barrier()


def run_convert_checkpoint_by_layout(src_in_strategy, dst_in_strategy, enable_parallel_optimizer,
                                     optimizer_weight_shard_size=None, use_safetensor=False):
    """
    1. run src network, save src strategy and src network checkpoint.
    2. run dst network, save dst strategy.
    3. convert src checkpoint to dst strategy.
    4. load converted checkpoints, compare with dst network params.
    """
    context.set_context(device_target='Ascend',
                        mode=ms.context.GRAPH_MODE, jit_level='O0')
    init()
    root_path = './test_checkpoints_convert_by_layout'
    if enable_parallel_optimizer:
        if optimizer_weight_shard_size:
            root_path += '_with_opt_shard_size_{}'.format(
                optimizer_weight_shard_size)
        else:
            root_path += '_with_opt_shard'
    if use_safetensor:
        root_path += '_safetensor'
    src_strategy_path = os.path.join(root_path, 'src_stra')
    src_ckpt_path = os.path.join(root_path, 'src_ckpt')
    dst_strategy_path = os.path.join(root_path, 'dst_stra')
    dst_ckpt_path = os.path.join(root_path, 'dst_ckpt')

    in_channels = 8
    out_channels = 16
    bs = 8
    input_data = ms.Tensor(np.ones((bs, in_channels)), dtype=mstype.float32)

    src_network = TestNet(in_channels=in_channels, out_channels=out_channels,
                          in_strategy=src_in_strategy)
    compile_net_and_save_ckpt(src_network,
                              input_data,
                              enable_parallel_optimizer=enable_parallel_optimizer,
                              optimizer_weight_shard_size=optimizer_weight_shard_size,
                              save_ckpt_path=src_ckpt_path,
                              save_strategy_path=src_strategy_path,
                              use_safetensor=use_safetensor)

    dst_network = TestNet(in_channels=in_channels, out_channels=out_channels,
                          in_strategy=dst_in_strategy)
    compile_net_and_save_ckpt(dst_network,
                              input_data,
                              enable_parallel_optimizer=enable_parallel_optimizer,
                              optimizer_weight_shard_size=optimizer_weight_shard_size,
                              save_strategy_path=dst_strategy_path,
                              use_safetensor=use_safetensor)

    run_convert(
        src_strategy_file=os.path.join(src_strategy_path, 'strategy_0.ckpt'),
        dst_strategy_file=os.path.join(dst_strategy_path, 'strategy_0.ckpt'),
        src_ckpt_path=src_ckpt_path,
        dst_ckpt_path=dst_ckpt_path,
        use_safetensor=use_safetensor,
    )

    compare_ckpt_and_network_params(dst_ckpt_path, dst_network, use_safetensor)
    clean_ckpts([src_strategy_path, src_ckpt_path,
                 dst_strategy_path, dst_ckpt_path])


def test_checkpoints_convert_by_layout():
    """ test checkpoints convert using layout. """
    layout = Layout(device_matrix=(2, 2, 2), alias_name=('dp', 'sp', 'mp'))
    src_in_strategy = {
        'mul_0.weight': (layout('dp', ('sp', 'mp')), layout(('sp', 'mp'))),
        'matmul_0.weight': (layout('dp', 'sp'), layout('sp', 'mp')),
        'matmul_1.weight': (layout('dp', 'None'), layout('None', ('sp', 'mp'))),
        'add_0.weight': (layout('dp', 'mp'), layout('mp')),
    }
    dst_in_strategy = {
        'mul_0.weight': (layout('dp', 'mp'), layout('mp')),
        'matmul_0.weight': (layout(('dp', 'sp'), 'mp'), layout('mp', 'None')),
        'matmul_1.weight': (layout('dp', ('sp', 'mp')), layout(('sp', 'mp'), 'None')),
        'add_0.weight': (layout('dp', ('sp', 'mp')), layout(('sp', 'mp'))),
    }
    run_convert_checkpoint_by_layout(
        src_in_strategy=src_in_strategy,
        dst_in_strategy=dst_in_strategy,
        enable_parallel_optimizer=False,
    )


def test_checkpoints_convert_by_layout_with_opt_shard_safetensor():
    """ test checkpoints convert using layout with optimizer shard. """
    layout = Layout(device_matrix=(4, 2), alias_name=('dp', 'mp'))
    src_in_strategy = {
        'mul_0.weight': (layout('None', ('dp', 'mp')), layout(('dp', 'mp'))),
        'matmul_0.weight': (layout('dp', 'None'), layout('None', 'mp')),
        'matmul_1.weight': (layout('None', 'None'), layout('None', ('dp', 'mp'))),
        'add_0.weight': (layout('dp', 'mp'), layout('mp')),
    }
    dst_in_strategy = {
        'mul_0.weight': (layout('dp', 'mp'), layout('mp')),
        'matmul_0.weight': (layout(('dp', 'mp'), 'None'), layout('None', 'None')),
        'matmul_1.weight': (layout('None', ('dp', 'mp')), layout(('dp', 'mp'), 'None')),
        'add_0.weight': (layout(('dp', 'mp'), 'None'), layout('None')),
    }
    run_convert_checkpoint_by_layout(
        src_in_strategy=src_in_strategy,
        dst_in_strategy=dst_in_strategy,
        enable_parallel_optimizer=True,
        optimizer_weight_shard_size=2,
        use_safetensor=True,
    )
