# Copyright 2020 Huawei Technologies Co., Ltd
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
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.dataset as ds

from mindspore.communication.management import init
from mindspore import ops
from mindspore.train import LossMonitor
from mindspore.train import Model
from mindspore import lazy_inline
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.parallel.auto_parallel import AutoParallel
from mindspore.nn.utils import no_init_parameters
from mindspore.common.initializer import initializer, HeUniform
from mindspore.nn import PipelineCell
from .model_parallel import FakeData, FakeDataInitMode

context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')

class ParallelPPNetworkFirst(nn.Cell):
    """ParallelPPNetworkFirst"""
    def __init__(self, strategy=None):
        super().__init__()
        self.flatten = ops.Flatten()
        self.fc1_weight = ms.Parameter(initializer(HeUniform(math.sqrt(5)), shape=[16, 16], dtype=ms.float32),
                                       name="fc1")
        self.matmul = ops.MatMul().shard(strategy)
        self.relu = ops.ReLU()

    def construct(self, x):
        x = self.flatten(x)
        x = self.matmul(x, self.fc1_weight)
        x = self.relu(x)
        return x

class ParallelPPNetworkSecond(nn.Cell):
    """ParallelPPNetworkSecond"""
    def __init__(self, strategy=None):
        super().__init__()
        self.flatten = ops.Flatten()
        self.fc1_weight = ms.Parameter(initializer(HeUniform(math.sqrt(5)), shape=[16, 16], dtype=ms.float32),
                                       name="fc1")
        self.matmul = ops.MatMul()
        self.relu = ops.ReLU()

    def construct(self, x):
        x = self.flatten(x)
        x = self.matmul(x, self.fc1_weight)
        x = self.relu(x)
        return x

class ParallelPPNetworkFinal(nn.Cell):
    """ParallelPPNetworkFinal"""
    def __init__(self, strategy=None):
        super().__init__()
        self.cell1 = ParallelPPNetworkFirst(strategy)
        self.cell2 = ParallelPPNetworkSecond(strategy)
        self.relu = ops.ReLU()

    def construct(self, x):
        x = self.cell1(x)
        x = self.cell2(x)
        x = self.relu(x)
        return x

class LazyLossCell(nn.Cell):
    @lazy_inline
    def __init__(self, backbone, loss_fn):
        super(LazyLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn
        self._get_attr_from_cell(backbone)
    def construct(self, data, label):
        out = self._backbone(data)
        return self._loss_fn(out, label)
    @property
    def backbone_network(self):
        return self._backbone

def create_dataset(batch_size):
    class RandomAccessDataset:
        def __init__(self):
            self.dataset_size = 64
            self.size = 64
            self.rank_batch_size = batch_size
            self.total_batch_size = self.rank_batch_size * self.size
        def __getitem__(self, index):
            np.random.seed(1)
            image_np = np.random.randn(batch_size, 4, 4).astype(np.float32)
            label_np = np.random.randint(low=0, high=1, size=batch_size*4*4, dtype=np.int32)
            label_np = np.reshape(label_np * 0.001, (batch_size, 4*4))
            return ms.Tensor(image_np), ms.Tensor(label_np, dtype=ms.float32)
        def __len__(self):
            return self.dataset_size
        def __iter__(self):
            self.batch_index = 0
            return self
        def reset(self):
            self.batch_index = 0

        def __next__(self):
            if self.batch_index * self.total_batch_size < len(self):
                data = self[self.batch_index]
                self.batch_index += 1
                return data
            raise StopIteration

    loader = RandomAccessDataset()
    rank_id = int(os.environ.get('RANK_ID', -1))
    rank_size = 8
    return ds.GeneratorDataset(source=loader, column_names=["image", "label"], num_shards=rank_size,
                               shard_id=rank_id)

def clean_all_ckpt_files(folder_path):
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.ckpt') or file_name.endswith('.meta'):
                os.remove(os.path.join(folder_path, file_name))

def test_parallel_mp_compare_context_autoparallel_pipeline_config():
    """
    Feature:test_parallel_mp_compare_context_autoparallel_pipeline_config
    Description:
        1.create pp Net
        2.train the net, using new pipeline_stage config
        3.predict net
    Expectation:
        1.train ok
        2.the predcit result is the same
    """
    rank_id = int(os.environ.get('RANK_ID', -1))
    init(backend_name='hccl')
    pp_strategy = ((2, 1), (1, 2))
    # set_parallel_context
    ms.set_seed(1)
    context.reset_auto_parallel_context()
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', save_graphs=True,
                        save_graphs_path="context_ir")
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      dataset_strategy="full_batch", pipeline_stages=2)

    context_net = ParallelPPNetworkFinal(pp_strategy)
    context_dataset = FakeData(size=64, batch_size=8, image_size=(4, 4), num_classes=16,
                               fakedata_mode=FakeDataInitMode.UniqueInit)
    context_ckpt_path = 'context_ckpt/context_rank_{}_ckpt'.format(rank_id)
    context_net.cell1.pipeline_stage = 0
    context_net.cell2.pipeline_stage = 1
    # train_model
    optimizer = nn.Momentum(context_net.trainable_params(), learning_rate=0.1, momentum=0.9)
    loss_fn = nn.MSELoss(reduction='mean')
    loss_monitor = LossMonitor(per_print_times=1)
    clean_all_ckpt_files(context_ckpt_path)
    net_with_loss = PipelineCell(nn.WithLossCell(context_net, loss_fn), micro_size=4)
    # train WithLossCell
    model = Model(network=net_with_loss, optimizer=optimizer)
    model.train(epoch=2, train_dataset=context_dataset, dataset_sink_mode=False, callbacks=[loss_monitor])

    # AutoParallel(net) config
    ms.set_seed(1)
    context.reset_auto_parallel_context()
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', save_graphs=True,
                        save_graphs_path="parallel_ir")
    # dataset
    parallel_dataset = FakeData(size=64, batch_size=8, image_size=(4, 4), num_classes=16,
                                fakedata_mode=FakeDataInitMode.UniqueInit)
    parallel_ckpt_path = 'parallel_ckpt/parallel_rank_{}_ckpt'.format(rank_id)
    # net
    with no_init_parameters():
        net_tmp = ParallelPPNetworkFinal(pp_strategy)
        optimizer = nn.Momentum(net_tmp.trainable_params(), learning_rate=0.1, momentum=0.9)
    # train_model ops loss
    loss_fn = nn.MSELoss(reduction='mean')

    clean_all_ckpt_files(parallel_ckpt_path)
    pp_net_with_loss = PipelineCell(nn.WithLossCell(net_tmp, loss_fn), micro_size=4,
                                    stage_config={"_backbone.cell1": 0,
                                                  "_backbone.cell2": 1})
    pp_net = AutoParallel(pp_net_with_loss, parallel_mode="semi_auto")
    pp_net.full_batch = True
    pp_net.pipeline(stages=2, scheduler="1f1b")
    loss_monitor_pp = LossMonitor(per_print_times=1)
    # train WithLossCell
    model = Model(network=pp_net, optimizer=optimizer)
    model.train(epoch=2, train_dataset=parallel_dataset, dataset_sink_mode=False, callbacks=[loss_monitor_pp])

def test_parallel_mp_compare_context_auto_pp_config_lazy_init():
    """
    Feature:test_parallel_mp_compare_context_auto_pp_config_lazy_init
    Description:
        1.create pp Net
        2.train the net, using new pipeline_stage config & full_batch
        3.predict net
    Expectation:
        1.train ok
        2.the predcit result is the same
    """
    rank_id = int(os.environ.get('RANK_ID', -1))
    init(backend_name='hccl')
    pp_strategy = ((2, 1), (1, 2))
    # set_parallel_context
    ms.set_seed(1)
    context.reset_auto_parallel_context()
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', save_graphs=True,
                        save_graphs_path="context_ir")
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      dataset_strategy="full_batch", pipeline_stages=2)

    context_net = ParallelPPNetworkFinal(pp_strategy)
    context_dataset = FakeData(size=64, batch_size=8, image_size=(4, 4), num_classes=16,
                               fakedata_mode=FakeDataInitMode.UniqueInit)
    context_ckpt_path = 'context_ckpt/context_rank_{}_ckpt'.format(rank_id)
    context_net.cell1.pipeline_stage = 0
    context_net.cell2.pipeline_stage = 1
    # train_model
    optimizer = nn.Momentum(context_net.trainable_params(), learning_rate=0.1, momentum=0.9)
    loss_fn = nn.MSELoss(reduction='mean')
    loss_monitor = LossMonitor(per_print_times=1)
    clean_all_ckpt_files(context_ckpt_path)
    net_with_loss = PipelineCell(nn.WithLossCell(context_net, loss_fn), micro_size=4)
    # train WithLossCell
    model = Model(network=net_with_loss, optimizer=optimizer)
    model.train(epoch=2, train_dataset=context_dataset, dataset_sink_mode=False, callbacks=[loss_monitor])
    # AutoParallel(net)
    # config
    ms.set_seed(1)
    context.reset_auto_parallel_context()
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', save_graphs=True,
                        save_graphs_path="parallel_ir")
    # dataset
    parallel_dataset = FakeData(size=64, batch_size=8, image_size=(4, 4), num_classes=16,
                                fakedata_mode=FakeDataInitMode.UniqueInit)
    parallel_ckpt_path = 'parallel_ckpt/parallel_rank_{}_ckpt'.format(rank_id)
    # net
    with no_init_parameters():
        net_tmp = ParallelPPNetworkFinal(pp_strategy)
        optimizer = nn.Momentum(net_tmp.trainable_params(), learning_rate=0.1, momentum=0.9)
    # train_model ops loss
    loss_fn = nn.MSELoss(reduction='mean')

    clean_all_ckpt_files(parallel_ckpt_path)
    pp_net_with_loss = PipelineCell(nn.WithLossCell(net_tmp, loss_fn), micro_size=4,
                                    stage_config={"_backbone.cell1": 0,
                                                  "_backbone.cell2": 1})
    pp_net = AutoParallel(pp_net_with_loss, parallel_mode="semi_auto")
    pp_net.full_batch = True
    pp_net.pipeline(stages=2, scheduler="1f1b")
    loss_monitor_pp = LossMonitor(per_print_times=1)
    # train WithLossCell
    model = Model(network=pp_net, optimizer=optimizer)
    model.train(epoch=2, train_dataset=parallel_dataset, dataset_sink_mode=False,
                callbacks=[loss_monitor_pp])

def test_parallel_mp_compare_context_auto_pp_config_lazy_init_dp():
    """
    Feature:test_parallel_mp_compare_context_auto_pp_config_lazy_init_dp
    Description:
        1.create pp Net
        2.train the net, using new pipeline_stage config & data_parallel
        3.predict net
    Expectation:
        1.train ok
        2.the predcit result is the same
    """
    rank_id = int(os.environ.get('RANK_ID', -1))
    init(backend_name='hccl')
    pp_strategy = ((2, 1), (1, 2))
    # set_parallel_context
    ms.set_seed(1)
    context.reset_auto_parallel_context()
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', save_graphs=True,
                        save_graphs_path="context_ir")
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      dataset_strategy="data_parallel", pipeline_stages=2)
    context_net = ParallelPPNetworkFinal(pp_strategy)
    context_dataset = FakeData(size=64, batch_size=8, image_size=(4, 4), num_classes=16,
                               fakedata_mode=FakeDataInitMode.UniqueInit)
    context_ckpt_path = 'context_ckpt/context_rank_{}_ckpt'.format(rank_id)
    context_net.cell1.pipeline_stage = 0
    context_net.cell2.pipeline_stage = 1
    # train_model
    optimizer = nn.Momentum(context_net.trainable_params(), learning_rate=0.1, momentum=0.9)
    loss_fn = nn.MSELoss(reduction='mean')
    loss_monitor = LossMonitor(per_print_times=1)
    clean_all_ckpt_files(context_ckpt_path)
    net_with_loss = PipelineCell(nn.WithLossCell(context_net, loss_fn), micro_size=4)
    # train WithLossCell
    model = Model(network=net_with_loss, optimizer=optimizer)
    model.train(epoch=2, train_dataset=context_dataset, dataset_sink_mode=False, callbacks=[loss_monitor])
    # AutoParallel(net)
    # config
    ms.set_seed(1)
    context.reset_auto_parallel_context()
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', save_graphs=True,
                        save_graphs_path="parallel_ir")
    # dataset
    parallel_dataset = FakeData(size=64, batch_size=8, image_size=(4, 4), num_classes=16,
                                fakedata_mode=FakeDataInitMode.UniqueInit)
    parallel_ckpt_path = 'parallel_ckpt/parallel_rank_{}_ckpt'.format(rank_id)
    # net
    with no_init_parameters():
        net_tmp = ParallelPPNetworkFinal(pp_strategy)
        optimizer = nn.Momentum(net_tmp.trainable_params(), learning_rate=0.1, momentum=0.9)
    # train_model ops loss
    loss_fn = nn.MSELoss(reduction='mean')
    clean_all_ckpt_files(parallel_ckpt_path)
    pp_net_with_loss = PipelineCell(nn.WithLossCell(net_tmp, loss_fn), micro_size=4,
                                    stage_config={"_backbone.cell1": 0,
                                                  "_backbone.cell2": 1})
    pp_net = AutoParallel(pp_net_with_loss, parallel_mode="semi_auto")
    pp_net.full_batch = True
    pp_net.pipeline(stages=2, scheduler="1f1b")
    loss_monitor_pp = LossMonitor(per_print_times=1)
    # train WithLossCell
    model = Model(network=pp_net, optimizer=optimizer)
    model.train(epoch=2, train_dataset=parallel_dataset, dataset_sink_mode=False,
                callbacks=[loss_monitor_pp])


def test_parallel_mp_compare_context_auto_pp_config_with_lazy_init_inline():
    """
    Feature:test_parallel_mp_compare_context_autoparallel_pipeline_config_with_lazy_init_lazy_inline
    Description:
        1.create pp Net
        2.train the net, using new pipeline_stage config
        3.predict net
    Expectation:
        1.train ok
        2.the predcit result is the same
    """
    rank_id = int(os.environ.get('RANK_ID', -1))
    init(backend_name='hccl')
    pp_strategy = ((2, 1), (1, 2))
    # set_parallel_context
    ms.set_seed(1)
    context.reset_auto_parallel_context()
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend',
                        save_graphs=True, save_graphs_path="context_ir")
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      dataset_strategy="full_batch", pipeline_stages=2)
    context_net = ParallelPPNetworkFinal(pp_strategy)
    context_dataset = FakeData(size=64, batch_size=8, image_size=(4, 4), num_classes=16,
                               fakedata_mode=FakeDataInitMode.UniqueInit)
    context_ckpt_path = 'context_ckpt/context_rank_{}_ckpt'.format(rank_id)
    context_net.cell1.pipeline_stage = 0
    context_net.cell2.pipeline_stage = 1
    # train_model
    optimizer = nn.Momentum(context_net.trainable_params(), learning_rate=0.1, momentum=0.9)
    loss_fn = nn.MSELoss(reduction='mean')
    loss_monitor = LossMonitor(per_print_times=1)
    clean_all_ckpt_files(context_ckpt_path)
    # loss with lazy inline
    net_with_loss = PipelineCell(LazyLossCell(context_net, loss_fn), micro_size=4)
    # train WithLossCell
    model = Model(network=net_with_loss, optimizer=optimizer)
    model.train(epoch=2, train_dataset=context_dataset, dataset_sink_mode=False, callbacks=[loss_monitor])
    # AutoParallel(net)
    # config
    ms.set_seed(1)
    context.reset_auto_parallel_context()
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', save_graphs=True,
                        save_graphs_path="parallel_ir")
    # dataset
    parallel_dataset = FakeData(size=64, batch_size=8, image_size=(4, 4), num_classes=16,
                                fakedata_mode=FakeDataInitMode.UniqueInit)
    parallel_ckpt_path = 'parallel_ckpt/parallel_rank_{}_ckpt'.format(rank_id)
    # net
    with no_init_parameters():
        net_tmp = ParallelPPNetworkFinal(pp_strategy)
        optimizer = nn.Momentum(net_tmp.trainable_params(), learning_rate=0.1, momentum=0.9)
    # train_model ops loss
    loss_fn = nn.MSELoss(reduction='mean')

    clean_all_ckpt_files(parallel_ckpt_path)
    pp_net_with_loss = PipelineCell(nn.WithLossCell(net_tmp, loss_fn), micro_size=4,
                                    stage_config={"_backbone.cell1": 0,
                                                  "_backbone.cell2": 1})
    pp_net = AutoParallel(pp_net_with_loss, parallel_mode="semi_auto")
    pp_net.full_batch = True
    pp_net.pipeline(stages=2, scheduler="1f1b")
    loss_monitor_pp = LossMonitor(per_print_times=1)
    # train WithLossCell
    model = Model(network=pp_net, optimizer=optimizer)
    model.train(epoch=2, train_dataset=parallel_dataset, dataset_sink_mode=False, callbacks=[loss_monitor_pp])

def test_parallel_mp_compare_context_auto_pp_cfg_lazy_init_inline_sink():
    """
    Feature:test_parallel_mp_compare_context_auto_pp_config_lazy_init
    Description:
        1.create pp Net
        2.train the net, using new pipeline_stage config
        3.predict net
    Expectation:
        1.train ok
        2.the predcit result is the same
    """
    rank_id = int(os.environ.get('RANK_ID', -1))
    init(backend_name='hccl')
    pp_strategy = ((2, 1), (1, 2))
    # set_parallel_context
    # config
    ms.set_seed(1)
    context.reset_auto_parallel_context()
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', save_graphs=True,
                        save_graphs_path="context_ir")
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      dataset_strategy="full_batch", pipeline_stages=2)
    # net
    context_net = ParallelPPNetworkFinal(pp_strategy)
    # dataset
    context_dataset = create_dataset(8)
    context_ckpt_path = 'context_ckpt/context_rank_{}_ckpt'.format(rank_id)
    # pp config
    context_net.cell1.pipeline_stage = 0
    context_net.cell2.pipeline_stage = 1
    # train_model ops loss
    optimizer = nn.Momentum(context_net.trainable_params(), learning_rate=0.1, momentum=0.9)
    loss_fn = nn.MSELoss(reduction='mean')
    loss_monitor = LossMonitor(per_print_times=1)
    clean_all_ckpt_files(context_ckpt_path)
    # loss with lazy inline
    net_with_loss = PipelineCell(LazyLossCell(context_net, loss_fn), micro_size=4)
    # train WithLossCell
    model = Model(network=net_with_loss, optimizer=optimizer)
    model.train(epoch=2, train_dataset=context_dataset, dataset_sink_mode=True, sink_size=-1, callbacks=[loss_monitor])
    # AutoParallel(net)
    # config
    ms.set_seed(1)
    context.reset_auto_parallel_context()
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend',
                        save_graphs=True, save_graphs_path="parallel_ir")
    # dataset
    parallel_dataset = create_dataset(8)
    parallel_ckpt_path = 'parallel_ckpt/parallel_rank_{}_ckpt'.format(rank_id)
    # net
    with no_init_parameters():
        net_tmp = ParallelPPNetworkFinal(pp_strategy)
        optimizer = nn.Momentum(net_tmp.trainable_params(), learning_rate=0.1, momentum=0.9)
    # train_model ops loss
    loss_fn = nn.MSELoss(reduction='mean')
    loss_monitor_pp = LossMonitor(per_print_times=1)
    clean_all_ckpt_files(parallel_ckpt_path)
    # loss with lazy inline & pp config
    pp_net_with_loss = PipelineCell(LazyLossCell(net_tmp, loss_fn),
                                    micro_size=4,
                                    stage_config={"_backbone.cell1": 0,
                                                  "_backbone.cell2": 1})
    pp_net = AutoParallel(pp_net_with_loss, parallel_mode="semi_auto")
    pp_net.full_batch = True
    pp_net.pipeline(stages=2, scheduler="1f1b")
    # train WithLossCell
    model = Model(network=pp_net, optimizer=optimizer)
    model.train(epoch=2, train_dataset=parallel_dataset, dataset_sink_mode=True, sink_size=-1,
                callbacks=[loss_monitor_pp])

def test_parallel_mp_compare_context_auto_sink_gpipe():
    """
    Feature:test_parallel_mp_compare_context_auto_pp_config_lazy_init
    Description:
        1.create pp Net
        2.train the net, using new pipeline_stage config
        3.predict net
    Expectation:
        1.train ok
        2.the predcit result is the same
    """
    rank_id = int(os.environ.get('RANK_ID', -1))
    init(backend_name='hccl')
    pp_strategy = ((2, 1), (1, 2))
    # set_parallel_context
    # config
    ms.set_seed(1)
    context.reset_auto_parallel_context()
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', save_graphs=True,
                        save_graphs_path="context_ir")
    pp_config = {"pipeline_interleave": True, "pipeline_scheduler": "gpipe"}
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      dataset_strategy="full_batch", pipeline_stages=2,
                                      pipeline_config=pp_config)
    # net
    context_net = ParallelPPNetworkFinal(pp_strategy)
    # dataset
    context_dataset = create_dataset(8)
    context_ckpt_path = 'context_ckpt/context_rank_{}_ckpt'.format(rank_id)
    # pp config
    context_net.cell1.pipeline_stage = 0
    context_net.cell2.pipeline_stage = 1
    # train_model ops loss
    optimizer = nn.Momentum(context_net.trainable_params(), learning_rate=0.1, momentum=0.9)
    loss_fn = nn.MSELoss(reduction='mean')
    loss_monitor = LossMonitor(per_print_times=1)
    clean_all_ckpt_files(context_ckpt_path)
    # loss with lazy inline
    net_with_loss = PipelineCell(LazyLossCell(context_net, loss_fn), micro_size=4)
    # train WithLossCell
    model = Model(network=net_with_loss, optimizer=optimizer)
    model.train(epoch=2, train_dataset=context_dataset, dataset_sink_mode=True, sink_size=-1, callbacks=[loss_monitor])
    # AutoParallel(net)
    # config
    ms.set_seed(1)
    context.reset_auto_parallel_context()
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend',
                        save_graphs=True, save_graphs_path="parallel_ir")
    # dataset
    parallel_dataset = create_dataset(8)
    parallel_ckpt_path = 'parallel_ckpt/parallel_rank_{}_ckpt'.format(rank_id)
    # net
    with no_init_parameters():
        net_tmp = ParallelPPNetworkFinal(pp_strategy)
        optimizer = nn.Momentum(net_tmp.trainable_params(), learning_rate=0.1, momentum=0.9)
    # train_model ops loss
    loss_fn = nn.MSELoss(reduction='mean')
    loss_monitor_pp = LossMonitor(per_print_times=1)
    clean_all_ckpt_files(parallel_ckpt_path)
    # loss with lazy inline & pp config
    pp_net_with_loss = PipelineCell(LazyLossCell(net_tmp, loss_fn),
                                    micro_size=4,
                                    stage_config={"_backbone.cell1": 0,
                                                  "_backbone.cell2": 1})
    pp_net = AutoParallel(pp_net_with_loss, parallel_mode="semi_auto")
    pp_net.full_batch = True
    pp_net.pipeline(stages=2, interleave=True, scheduler="gpipe")
    # train WithLossCell
    model = Model(network=pp_net, optimizer=optimizer)
    model.train(epoch=2, train_dataset=parallel_dataset, dataset_sink_mode=True, sink_size=-1,
                callbacks=[loss_monitor_pp])

def test_parallel_mp_compare_context_auto_sink_seqpipe():
    """
    Feature:test_parallel_mp_compare_context_auto_pp_config_lazy_init
    Description:
        1.create pp Net
        2.train the net, using new pipeline_stage config
        3.predict net
    Expectation:
        1.train ok
        2.the predcit result is the same
    """
    rank_id = int(os.environ.get('RANK_ID', -1))
    init(backend_name='hccl')
    pp_strategy = ((2, 1), (1, 2))
    # set_parallel_context
    # config
    ms.set_seed(1)
    context.reset_auto_parallel_context()
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', save_graphs=True,
                        save_graphs_path="context_ir")
    pp_config = {"pipeline_interleave": True, "pipeline_scheduler": "seqpipe"}
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      dataset_strategy="full_batch", pipeline_stages=2,
                                      pipeline_config=pp_config)
    # net
    context_net = ParallelPPNetworkFinal(pp_strategy)
    # dataset
    context_dataset = create_dataset(8)
    context_ckpt_path = 'context_ckpt/context_rank_{}_ckpt'.format(rank_id)
    # pp config
    context_net.cell1.pipeline_stage = 0
    context_net.cell2.pipeline_stage = 1
    # train_model ops loss
    optimizer = nn.Momentum(context_net.trainable_params(), learning_rate=0.1, momentum=0.9)
    loss_fn = nn.MSELoss(reduction='mean')
    loss_monitor = LossMonitor(per_print_times=1)
    clean_all_ckpt_files(context_ckpt_path)
    # loss with lazy inline
    net_with_loss = PipelineCell(LazyLossCell(context_net, loss_fn), micro_size=4)
    # train WithLossCell
    model = Model(network=net_with_loss, optimizer=optimizer)
    model.train(epoch=2, train_dataset=context_dataset, dataset_sink_mode=True, sink_size=-1, callbacks=[loss_monitor])
    # AutoParallel(net)
    # config
    ms.set_seed(1)
    context.reset_auto_parallel_context()
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend',
                        save_graphs=True, save_graphs_path="parallel_ir")
    # dataset
    parallel_dataset = create_dataset(8)
    parallel_ckpt_path = 'parallel_ckpt/parallel_rank_{}_ckpt'.format(rank_id)
    # net
    with no_init_parameters():
        net_tmp = ParallelPPNetworkFinal(pp_strategy)
        optimizer = nn.Momentum(net_tmp.trainable_params(), learning_rate=0.1, momentum=0.9)
    # train_model ops loss
    loss_fn = nn.MSELoss(reduction='mean')
    loss_monitor_pp = LossMonitor(per_print_times=1)
    clean_all_ckpt_files(parallel_ckpt_path)
    # loss with lazy inline & pp config
    pp_net_with_loss = PipelineCell(LazyLossCell(net_tmp, loss_fn),
                                    micro_size=4,
                                    stage_config={"_backbone.cell1": 0,
                                                  "_backbone.cell2": 1})
    pp_net = AutoParallel(pp_net_with_loss, parallel_mode="semi_auto")
    pp_net.full_batch = True
    pp_net.pipeline(stages=2, interleave=True, scheduler="seqpipe")
    # train WithLossCell
    model = Model(network=pp_net, optimizer=optimizer)
    model.train(epoch=2, train_dataset=parallel_dataset, dataset_sink_mode=True, sink_size=-1,
                callbacks=[loss_monitor_pp])
