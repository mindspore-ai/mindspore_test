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
import shutil
import numpy as np
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import context, Tensor, Parameter
from mindspore.nn import Cell, Momentum
from mindspore.ops import operations as P
from mindspore.communication.management import init, get_rank
from mindspore.parallel.shard import Layout
from mindspore.train import Model

class TestNet(Cell):
    def __init__(self, in_channels, out_channels, use_parallel=True, in_strategy=None, out_strategy=None):
        super().__init__()
        # init operations
        mul_0_weight_np = np.arange(in_channels, dtype=np.float32)
        self.mul_0_weight = Parameter(Tensor(mul_0_weight_np, dtype=mstype.float32), name='mul_0.weight')
        self.mul_0 = P.Mul()

        matmul_0_weight_np = np.arange(in_channels * out_channels,
                                       dtype=np.float32).reshape((in_channels, out_channels))
        self.matmul_0_weight = Parameter(Tensor(matmul_0_weight_np, dtype=mstype.float32), name='matmul_0.weight')
        self.matmul_0 = P.MatMul(transpose_a=False, transpose_b=False)

        matmul_1_weight_np = np.arange(out_channels * out_channels,
                                       dtype=np.float32).reshape((out_channels, out_channels))
        self.matmul_1_weight = Parameter(Tensor(matmul_1_weight_np, dtype=mstype.float32), name='matmul_1.weight')
        self.matmul_1 = P.MatMul(transpose_a=False, transpose_b=False)

        add_0_weight_np = np.arange(out_channels, dtype=np.float32)
        self.add_0_weight = Parameter(Tensor(add_0_weight_np, dtype=mstype.float32), name='add_0.weight')
        self.add_0 = P.Add()

        # set_layout
        if use_parallel:
            if not in_strategy:
                in_strategy = dict()
            if not out_strategy:
                out_strategy = dict()
            self.mul_0.shard(in_strategy=in_strategy.get(self.mul_0_weight.name),
                             out_strategy=out_strategy.get(self.mul_0_weight.name))
            self.matmul_0.shard(in_strategy=in_strategy.get(self.matmul_0_weight.name),
                                out_strategy=out_strategy.get(self.matmul_0_weight.name))
            self.matmul_1.shard(in_strategy=in_strategy.get(self.matmul_1_weight.name),
                                out_strategy=out_strategy.get(self.matmul_1_weight.name))
            self.add_0.shard(in_strategy=in_strategy.get(self.add_0_weight.name),
                             out_strategy=out_strategy.get(self.add_0_weight.name))

    def construct(self, input_):
        out = self.mul_0(input_, self.mul_0_weight)
        out = self.matmul_0(out, self.matmul_0_weight)
        out = self.matmul_1(out, self.matmul_1_weight)
        out = self.add_0(out, self.add_0_weight)
        return out


def compile_net_and_save_ckpt(network, input_data, enable_parallel_optimizer=False, optimizer_weight_shard_size=None,
                              save_ckpt_path=None, save_strategy_path=None, use_safetensor=False):
    ms.reset_auto_parallel_context()
    parallel_optimizer_config = {'parallel_optimizer_threshold': 0}
    if optimizer_weight_shard_size:
        parallel_optimizer_config['optimizer_weight_shard_size'] = optimizer_weight_shard_size
    ms.set_auto_parallel_context(
        dataset_strategy='full_batch',
        parallel_mode='semi_auto_parallel',
        enable_parallel_optimizer=enable_parallel_optimizer,
        parallel_optimizer_config=parallel_optimizer_config,
    )

    if save_strategy_path:
        ms.set_auto_parallel_context(
            strategy_ckpt_config={
                'save_file': os.path.join(save_strategy_path, 'strategy_{}.ckpt'.format(get_rank()))
            }
        )

    optimizer = Momentum(learning_rate=0.0, momentum=0.9, params=network.trainable_params())
    model = Model(network=network, optimizer=optimizer)
    model.infer_predict_layout(input_data)
    ckpt_format = 'safetensors' if use_safetensor else 'ckpt'

    if save_ckpt_path:
        if not os.path.exists(os.path.join(save_ckpt_path, 'rank_{}'.format(get_rank()))):
            os.makedirs(os.path.join(save_ckpt_path, 'rank_{}'.format(get_rank())), exist_ok=True)
        ms.save_checkpoint(
            network,
            os.path.join(save_ckpt_path, 'rank_{}/checkpoint_{}.{}'.format(get_rank(), get_rank(), ckpt_format)),
            integrated_save=False, format=ckpt_format
        )
    ms.mint.distributed.barrier()


def run_transform(src_strategy_file, dst_strategy_file, src_ckpt_path, dst_ckpt_path, use_safetensor=False):
    target_rank = get_rank()
    if use_safetensor:
        if get_rank() == 0:
            ms.transform_checkpoints(src_ckpt_path, dst_ckpt_path, 'checkpoint_',
                                     src_strategy_file, dst_strategy_file, output_format='safetensors')
    else:
        rank_list = ms.rank_list_for_transform(target_rank, src_strategy_file, dst_strategy_file)
        checkpoint_files_map = {}
        for rank in rank_list:
            checkpoint_files_map[rank] = "{path}/rank_{rank}/checkpoint_{rank}.ckpt".format(
                path=src_ckpt_path, rank=rank)
        save_checkpoint_file_name = os.path.join(dst_ckpt_path, "rank_{rank}/checkpoint_{rank}.ckpt".
                                                 format(rank=target_rank))
        if not os.path.exists(os.path.join(dst_ckpt_path, 'rank_{}'.format(get_rank()))):
            os.makedirs(os.path.join(dst_ckpt_path, 'rank_{}'.format(get_rank())), exist_ok=True)
        ms.transform_checkpoint_by_rank(target_rank, checkpoint_files_map, save_checkpoint_file_name,
                                        src_strategy_file, dst_strategy_file)
    ms.mint.distributed.barrier()


def compare_ckpt_and_network_params(ckpt_path, network, use_safetensor=False):
    """compare checkpoint with network parameters."""
    ckpt_format = 'safetensors' if use_safetensor else 'ckpt'
    state_dict = ms.load_checkpoint(
        os.path.join(ckpt_path, 'rank_{}/checkpoint_{}.{}'.format(get_rank(), get_rank(), ckpt_format)),
        format=ckpt_format
    )
    for param in network.trainable_params():
        assert param.name in state_dict.keys()
        assert param.value().shape == state_dict.get(param.name).shape
        assert ms.mint.equal(param.value(), state_dict.get(param.name).value())


def clean_ckpts(path_list):
    """clean checkpoint files"""
    for path in path_list:
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
    ms.mint.distributed.barrier()


def run_transform_checkpoint_by_layout(src_in_strategy, dst_in_strategy, enable_parallel_optimizer,
                                       optimizer_weight_shard_size=None, use_safetensor=False):
    """
    1. run src network, save src strategy and src network checkpoint.
    2. run dst network, save dst strategy.
    3. transform src checkpoint to dst strategy.
    4. load transformed checkpoints, compare with dst network params.
    """
    context.set_context(device_target='Ascend', mode=ms.context.GRAPH_MODE, jit_level='O0')
    init()
    root_path = './test_checkpoints_transform_by_layout'
    if enable_parallel_optimizer:
        if optimizer_weight_shard_size:
            root_path += '_with_opt_shard_size_{}'.format(optimizer_weight_shard_size)
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

    run_transform(
        src_strategy_file=os.path.join(src_strategy_path, 'strategy_0.ckpt'),
        dst_strategy_file=os.path.join(dst_strategy_path, 'strategy_0.ckpt'),
        src_ckpt_path=src_ckpt_path,
        dst_ckpt_path=dst_ckpt_path,
        use_safetensor=use_safetensor,
    )

    compare_ckpt_and_network_params(dst_ckpt_path, dst_network, use_safetensor)
    clean_ckpts([src_strategy_path, src_ckpt_path, dst_strategy_path, dst_ckpt_path])


def test_checkpoints_transform_by_layout():
    """ test checkpoints transform using layout. """
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
    run_transform_checkpoint_by_layout(
        src_in_strategy=src_in_strategy,
        dst_in_strategy=dst_in_strategy,
        enable_parallel_optimizer=False,
    )


def test_checkpoints_transform_by_layout_with_opt_shard():
    """ test checkpoints transform using layout with optimizer shard. """
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
    run_transform_checkpoint_by_layout(
        src_in_strategy=src_in_strategy,
        dst_in_strategy=dst_in_strategy,
        enable_parallel_optimizer=True,
        optimizer_weight_shard_size=2,
    )


def test_checkpoints_transform_by_layout_with_opt_shard_safetensor():
    """ test checkpoints transform using layout with optimizer shard. """
    layout = Layout(device_matrix=(4, 1, 2), alias_name=('dp', 'one', 'mp'))
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
    run_transform_checkpoint_by_layout(
        src_in_strategy=src_in_strategy,
        dst_in_strategy=dst_in_strategy,
        enable_parallel_optimizer=True,
        optimizer_weight_shard_size=2,
        use_safetensor=True,
    )
