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

import sys
import numpy as np
import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn, amp
from mindspore.train import Model
from mindspore.common.initializer import Normal
from mindspore.ops import operations as P
from mindspore.communication import init, get_rank, get_group_size

ms.set_context(max_device_memory="2GB")
ms.set_context(mode=ms.GRAPH_MODE, jit_level='O0')
ms.set_context(save_graphs=True, save_graphs_path='ms_graphs')
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
init()
ms.set_seed(1)
np.random.seed(1)

class LeNet5(nn.Cell):
    """
    Lenet network

    Args:
        num_class (int): Number of classes. Default: 10.
        num_channel (int): Number of channels. Default: 1.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> LeNet5(num_class=10)

    """
    def __init__(self, num_class=10, num_channel=1, use_fp16_weight=False, use_fp16_input=False):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reshape = P.Reshape()
        weight_dtype = ms.float16 if use_fp16_weight else ms.float32
        self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02), dtype=weight_dtype)
        self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))
        self.cast = P.Cast()

    def construct(self, x):
        '''
        Forward network.
        '''
        x = self.cast(x, ms.float32)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.reshape(x, (-1, 400))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def create_dataset(batch_size, data_type):
    # """create dataset"""
    # Random-accessible object as input source
    class RandomAccessDataset:
        def __init__(self):
            self.dataset_size = 20

        def __getitem__(self, index):
            image_np = np.random.randn(batch_size, 1, 32, 32).astype(data_type)
            label_np = np.random.randint(low=0, high=10, size=batch_size, dtype=np.int32)
            return ms.Tensor(image_np), ms.Tensor(label_np)

        def __len__(self):
            return self.dataset_size

    loader = RandomAccessDataset()
    rank_id = get_rank()
    rank_size = get_group_size()
    return ds.GeneratorDataset(source=loader, column_names=["image", "label"], num_shards=rank_size, shard_id=rank_id)


def functional_train(data_set, net):
    optimizer = nn.SGD(net.trainable_params(), 1e-2)
    loss_fn = nn.CrossEntropyLoss()

    def forward_fn(data, target):
        """forward propagation"""
        logits = net(data)
        loss = loss_fn(logits, target)
        return loss, logits

    grad_fn = ms.value_and_grad(forward_fn, None, net.trainable_params(), has_aux=True)

    @ms.jit
    def train_step(inputs, targets):
        """train_step"""
        (loss_value, _), grads = grad_fn(inputs, targets)
        optimizer(grads)
        return loss_value

    for epoch in range(1):
        i = 0
        for image, label in data_set:
            loss_output = train_step(image, label)
            if i % 10 == 0:
                print("epoch: %s, step: %s, loss is %s" % (epoch, i, loss_output))
            i += 1


def model_train(data_set, net):
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    loss_scale_manager = ms.FixedLossScaleManager(1024., False)
    optim = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
    model = Model(net, loss_fn=loss, optimizer=optim, metrics=None, loss_scale_manager=loss_scale_manager)
    model.train(1, data_set, dataset_sink_mode=True, sink_size=1)


if __name__ == '__main__':
    fp16_type = None
    if len(sys.argv) > 1:
        fp16_type = sys.argv[1]
    print(f'fp16_type={fp16_type}')
    # "fp16_weight", "fp16_input", "fp16_getnext"
    is_fp16_weight = (fp16_type == "fp16_weight")
    is_fp16_input = (fp16_type == "fp16_input")
    is_fp16_getnext = (fp16_type == "fp16_getnext")

    has_fp16_input = is_fp16_input or is_fp16_getnext
    dtype = np.float16 if has_fp16_input else np.float32

    dataset = create_dataset(batch_size=32, data_type=dtype)
    network = LeNet5(use_fp16_weight=is_fp16_weight, use_fp16_input=has_fp16_input)
    if is_fp16_weight:
        network = amp.auto_mixed_precision(network, 'O1')
    if is_fp16_input:
        functional_train(dataset, network)
    else:
        model_train(dataset, network)
