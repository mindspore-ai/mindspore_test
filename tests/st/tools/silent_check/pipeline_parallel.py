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
"""SilentCheck based on Distributed Pipeline Parallel"""
import math
import numpy as np
import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn, ops, Parameter
from mindspore.train import Model
from mindspore.communication import init, get_rank, get_group_size
from mindspore.common.initializer import initializer, HeUniform

ms.set_context(mode=ms.GRAPH_MODE, jit_level='O0')
ms.set_context(save_graphs=True, save_graphs_path='ms_graphs')
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL, pipeline_stages=2)
init()
ms.set_seed(1)
np.random.seed(1)

class MatMulCell(nn.Cell):
    """
    MatMulCell definition.
    """
    def __init__(self, param=None, shape=None):
        super().__init__()
        if shape is None:
            shape = [28 * 28, 512]
        weight_init = HeUniform(math.sqrt(5))
        self.param = Parameter(initializer(weight_init, shape), name="param")
        if param is not None:
            self.param = param
        self.print = ops.Print()
        self.matmul = ops.MatMul()

    def construct(self, x):
        out = self.matmul(x, self.param)
        self.print("out is:", out)
        return out


class Network(nn.Cell):
    """
    Network definition.
    """
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = MatMulCell()
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Dense(512, 512)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Dense(512, 10)

    def construct(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        logits = self.layer3(x)
        return logits


def create_dataset(batch_size):
    # """create dataset"""
    # Random-accessible object as input source
    class RandomAccessDataset:
        def __init__(self):
            self.dataset_size = 1875

        def __getitem__(self, index):
            image_np = np.random.randn(batch_size, 1, 28, 28).astype(np.float32)
            label_np = np.random.randint(low=0, high=10, size=batch_size, dtype=np.int32)
            return ms.Tensor(image_np), ms.Tensor(label_np)

        def __len__(self):
            return self.dataset_size

    loader = RandomAccessDataset()
    rank_id = get_rank()
    rank_size = get_group_size()
    return ds.GeneratorDataset(source=loader, column_names=["image", "label"], num_shards=rank_size, shard_id=rank_id)


if __name__ == '__main__':
    dataset = create_dataset(batch_size=32)
    net = Network()
    net.layer1.pipeline_stage = 0
    net.relu1.pipeline_stage = 0
    net.layer2.pipeline_stage = 0
    net.relu2.pipeline_stage = 1
    net.layer3.pipeline_stage = 1

    optim = nn.SGD(net.trainable_params(), 1e-2)
    loss_fn = nn.CrossEntropyLoss()

    net_with_loss = nn.PipelineCell(nn.WithLossCell(net, loss_fn), 4)
    net_with_loss.set_train()

    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    optim = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
    model = Model(net_with_loss, optimizer=optim, metrics=None)
    model.train(1, dataset, dataset_sink_mode=True, sink_size=1)
