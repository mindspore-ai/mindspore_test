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

import mindspore as ms
from mindspore import nn
from mindspore.train import Model
import mindspore.dataset as ds
from mindspore.common.initializer import Normal
from mindspore.ops import operations as P
import numpy as np

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_context(max_device_memory="2GB")
ms.set_seed(1)
np.random.seed(1)

class LeNet5(nn.Cell):
    """
    Lenet network

    Args:
        num_class (int): Number of classes. Default: 10.
        num_channel (int): Number of channels. Default: 1.
        include_top (bool): If includes fc layers. Default: True.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> LeNet5(num_class=10)

    """
    def __init__(self, num_class=10, num_channel=1, include_top=True):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reshape = P.Reshape()
        # self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))
        # self.shape = Tensor(np.array([-1, 400]).astype(np.int64))


    def construct(self, x):
        '''
        Forward network.
        '''
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


def create_dataset(batch_size):
    # """create dataset"""
    # Random-accessible object as input source
    class RandomAccessDataset:
        def __init__(self):
            self.dataset_size = 20

        def __getitem__(self, index):
            image_np = np.random.randn(batch_size, 1, 32, 32).astype(np.float32)
            label_np = np.random.randint(low=0, high=10, size=batch_size, dtype=np.int32)
            return ms.Tensor(image_np), ms.Tensor(label_np)

        def __len__(self):
            return self.dataset_size

    loader = RandomAccessDataset()
    return ds.GeneratorDataset(source=loader, column_names=["image", "label"])


if __name__ == '__main__':
    ms.set_context(mode=ms.GRAPH_MODE, jit_level='O0')

    dataset = create_dataset(batch_size=32)
    net = LeNet5()
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    loss_scale_manager = ms.FixedLossScaleManager(1024., False)
    optim = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
    model = Model(net, loss_fn=loss, optimizer=optim, metrics=None, loss_scale_manager=loss_scale_manager)
    model.train(1, dataset, dataset_sink_mode=True, sink_size=1)
