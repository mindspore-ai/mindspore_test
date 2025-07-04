# Copyright 2022 Huawei Technologies Co., Ltd
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

import mindspore.common.dtype as mstype
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops.functional as F
from mindspore import Tensor, shard, set_seed
from mindspore.common.initializer import TruncatedNormal
from mindspore.communication.management import init
from mindspore.nn.loss.loss import LossBase
from mindspore.nn.optim.momentum import Momentum
from mindspore.ops import operations as P
from mindspore.train import Callback, Model
from mindspore.context import ParallelMode
import mindspore.dataset as ds
import mindspore.runtime as rt


def weight_variable():
    return TruncatedNormal(0.01)


def _conv3x3(in_channels, out_channels, stride=1, padding=0, pad_mode='same'):
    init_value = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=3, stride=stride, padding=padding, pad_mode=pad_mode, weight_init=init_value)


def _conv1x1(in_channels, out_channels, stride=1, padding=0, pad_mode='same'):
    init_value = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=1, stride=stride, padding=padding, pad_mode=pad_mode, weight_init=init_value)


def _conv7x7(in_channels, out_channels, stride=1, padding=0, pad_mode='same'):
    init_value = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=7, stride=stride, padding=padding, pad_mode=pad_mode, weight_init=init_value)


def _fused_bn(channels, momentum=0.9):
    return nn.BatchNorm2d(channels, momentum=momentum)


class BasicBlock(nn.Cell):
    expansion = 1

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 momentum=0.1):
        super(BasicBlock, self).__init__()

        self.conv1 = _conv3x3(in_channels, out_channels, stride=stride)
        self.conv2 = _conv3x3(out_channels, out_channels)
        self.relu = P.ReLU()
        self.down_sample_layer = None
        self.downsample = (in_channels != out_channels)
        if self.downsample:
            self.down_sample_layer = nn.SequentialCell([_conv1x1(in_channels,
                                                                 out_channels,
                                                                 stride=stride,
                                                                 padding=0),
                                                        _fused_bn(out_channels,
                                                                  momentum=momentum)])
        self.add = P.Add()

    def construct(self, x):
        identity = x

        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)

        if self.downsample:
            identity = self.down_sample_layer(identity)

        out = self.add(x, identity)
        out = self.relu(out)

        return out


class ResidualBlock(nn.Cell):
    expansion = 4

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1):
        super(ResidualBlock, self).__init__()

        out_chls = out_channels // self.expansion
        self.conv1 = _conv1x1(in_channels, out_chls, stride=1)

        self.conv2 = _conv3x3(out_chls, out_chls, stride=stride)

        self.conv3 = _conv1x1(out_chls, out_channels, stride=1)

        self.relu = P.ReLU()
        self.downsample = (in_channels != out_channels)
        self.stride = stride
        if self.downsample:
            self.conv_down_sample = _conv1x1(in_channels, out_channels,
                                             stride=stride)
        elif self.stride != 1:
            self.maxpool_down = nn.MaxPool2d(kernel_size=1, stride=2, pad_mode='same')

        self.add = P.Add()

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.downsample:
            identity = self.conv_down_sample(identity)
        elif self.stride != 1:
            identity = self.maxpool_down(identity)

        out = self.add(out, identity)
        out = self.relu(out)

        return out


class ResNet(nn.Cell):
    def __init__(self,
                 block,
                 layer_nums,
                 in_channels,
                 out_channels,
                 strides=None,
                 num_classes=100):
        super(ResNet, self).__init__()

        if strides is None:
            strides = [1, 2, 2, 2]
        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError("the length of "
                             "layer_num, inchannel, outchannel list must be 4!")

        self.conv1 = _conv7x7(3, 64, stride=2)
        self.relu = P.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')

        self.layer1 = self._make_layer(block,
                                       layer_nums[0],
                                       in_channel=in_channels[0],
                                       out_channel=out_channels[0],
                                       stride=strides[0])
        self.layer1_shard = self.layer1.shard(in_strategy=(None,), out_strategy=(None,))
        self.layer2 = self._make_layer(block,
                                       layer_nums[1],
                                       in_channel=in_channels[1],
                                       out_channel=out_channels[1],
                                       stride=strides[1])
        self.layer2_shard = self.layer2.shard(in_strategy=((1, 1, 1, 1),), out_strategy=(None,),
                                              parameter_plan={
                                                  'self.layer2.1.conv1.weight': (1, 8, 1, 1),
                                                  'self.layer2.0.conv_down_sample.weight': (8, 1, 1, 1),
                                              })
        self.layer3 = self._make_layer(block,
                                       layer_nums[2],
                                       in_channel=in_channels[2],
                                       out_channel=out_channels[2],
                                       stride=strides[2])
        self.layer3_shard = self.layer3.shard(in_strategy=((8, 1, 1, 1),), out_strategy=(None,))
        self.layer4 = self._make_layer(block,
                                       layer_nums[3],
                                       in_channel=in_channels[3],
                                       out_channel=out_channels[3],
                                       stride=strides[3])
        self.layer4_shard = shard(self.layer4, in_strategy=((8, 1, 1, 1),), out_strategy=(None,),
                                  parameter_plan={'self.layer4.0.conv2.weight': (8, 1, 1, 1)})

        self.mean = P.ReduceMean(keep_dims=True)
        self.end_point = nn.Dense(2048, num_classes, has_bias=True,
                                  weight_init=weight_variable(),
                                  bias_init=weight_variable()).add_flags_recursive(fp16=True)
        self.head = shard(self.end_point, in_strategy=((1, 8),), out_strategy=(None,))
        self.squeeze = P.Squeeze()
        self.cast = P.Cast()

    def _make_layer(self, block, layer_num, in_channel, out_channel, stride):
        layers = []
        resblk = block(in_channel, out_channel, stride=1)
        layers.append(resblk)

        for _ in range(1, layer_num - 1):
            resblk = block(out_channel, out_channel, stride=1)
            layers.append(resblk)

        resblk = block(out_channel, out_channel, stride=stride)
        layers.append(resblk)

        return nn.SequentialCell(layers)

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)

        c2 = self.layer1_shard(c1)
        c3 = self.layer2_shard(c2)
        c4 = self.layer3_shard(c3)
        c5 = self.layer4_shard(c4)

        out = self.mean(c5, (2, 3))
        out = self.squeeze(out)
        out = self.head(out)

        return out


def resnet50(class_num=10):
    return ResNet(ResidualBlock,
                  [1, 1, 2, 1],
                  [64, 256, 512, 1024],
                  [256, 512, 1024, 2048],
                  [2, 2, 2, 1],
                  class_num)


class SoftmaxCrossEntropyExpand(LossBase):
    def __init__(self, sparse=False):
        super(SoftmaxCrossEntropyExpand, self).__init__()
        self.exp = P.Exp()
        self.sum = P.ReduceSum(keep_dims=True)
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.div = P.Div()
        self.log = P.Log()
        self.sum_cross_entropy = P.ReduceSum(keep_dims=False)
        self.mul = P.Mul()
        self.mul2 = P.Mul()
        self.cast = P.Cast()
        self.mean = P.ReduceMean(keep_dims=False)
        self.sparse = sparse
        self.max = P.ReduceMax(keep_dims=True)
        self.sub = P.Sub()
        self.eps = Tensor(1e-24, mstype.float32)

    def construct(self, logit, label):
        logit = self.cast(logit, mstype.float32)
        logit_max = self.max(logit, -1)
        exp = self.exp(self.sub(logit, logit_max))
        exp_sum = self.sum(exp, -1)
        softmax_result = self.div(exp, exp_sum)
        if self.sparse:
            label = self.onehot(label, F.shape(logit)[1], self.on_value, self.off_value)

        softmax_result_log = self.log(softmax_result + self.eps)
        loss = self.sum_cross_entropy((self.mul(softmax_result_log, label)), -1)
        loss = self.mul2(F.scalar_to_tensor(-1.0), loss)
        loss = self.mean(loss, -1)

        return loss


rank_id = int(os.environ["RANK_ID"])
device_num = int(os.environ["RANK_SIZE"])


class DataGenerator():
    def __init__(self):
        self.index = 0

    def get_parallel_blocks(self, input_, strategy):
        blocks = [input_]
        i = 0
        for stra in strategy:
            temp = []
            while blocks:
                block = blocks.pop(0)
                temp.extend(np.split(block, stra, axis=i))
            blocks.extend(temp)
            i += 1
        self.index += 1
        return blocks

    def input_data(self, shape):
        data = (np.arange(np.prod(shape)).reshape(shape)).astype(np.float32)
        stra = [1] * len(shape)
        stra[0] = device_num
        data_parallel = self.get_parallel_blocks(data, stra)
        return Tensor(data), Tensor(data_parallel[rank_id])

    def label_data(self, shape):
        data = (np.arange(np.prod(shape)).reshape(shape) * 1000 / np.prod(shape)).astype(np.int32)
        stra = [1] * len(shape)
        stra[0] = device_num
        data_parallel = self.get_parallel_blocks(data, stra)
        return Tensor(data), Tensor(data_parallel[rank_id])


class Dataset():
    def __init__(self, predict, label, length=1, input_num=2, repeat_count=1):
        self.predict = predict
        self.label = label
        self.index = 0
        self.size = 1
        self.length = length
        self.input_num = input_num
        self.repeat_count = repeat_count

    def __getitem__(self, i):
        self.index = i
        if self.input_num == 2:
            return (self.predict, self.label)
        return (self.predict,)

    def __len__(self):
        return self.size

    def reset(self):
        self.index = 0

    def get_dataset_size(self):
        return self.length

    def get_repeat_count(self):
        return self.repeat_count


class ModelCallback(Callback):
    def __init__(self):
        super(ModelCallback, self).__init__()
        self.loss_list = []

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        result = cb_params.net_outputs
        self.loss_list.append(result.asnumpy().mean())


def train_feed(num_classes):
    parallel_callback = ModelCallback()
    data_gen = DataGenerator()
    _, input_part = data_gen.input_data((16 * 8, 3, 224, 224))
    _, label_part = data_gen.label_data((16 * 8,))
    dataset = Dataset(input_part, label_part)
    dataset = ds.GeneratorDataset(dataset, column_names=["image", "label"])
    net = resnet50(num_classes)
    loss = SoftmaxCrossEntropyExpand(sparse=True)
    loss.shard(in_strategy=(None, None), out_strategy=(None,))
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01, 0.9)
    model = Model(net, loss_fn=loss, optimizer=opt)
    model.train(2, dataset, dataset_sink_mode=False, callbacks=parallel_callback)


def test_train_feed_ascend():
    '''
    Feature: shard function for cell to enable parallel execution under PyNative mode in Ascend
    Description: Test a shrunk version of ResNet50 with a alternative execution of shard and pynative
    Expectation: Run success
    '''
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    rt.set_memory(max_size="25GB")
    context.set_context(device_id=int(os.getenv('DEVICE_ID')))
    init()
    context.set_auto_parallel_context(gradients_mean=True, parallel_mode=ParallelMode.AUTO_PARALLEL,
                                      search_mode="sharding_propagation", device_num=8,
                                      dataset_strategy="data_parallel")
    np.random.seed(42)
    set_seed(42)
    train_feed(num_classes=65536)


def test_train_feed_gpu():
    '''
    Feature: shard function for cell to enable parallel execution under PyNative mode in GPU
    Description: Test a shrunk version of ResNet50 with a alternative execution of shard and pynative
    Expectation: Run success
    '''
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    context.set_context(device_id=int(os.getenv('DEVICE_ID')))
    init()
    context.set_auto_parallel_context(gradients_mean=True, parallel_mode=ParallelMode.AUTO_PARALLEL,
                                      search_mode="sharding_propagation", device_num=8,
                                      dataset_strategy="data_parallel")
    np.random.seed(1)
    set_seed(1)
    train_feed(num_classes=65536)

def test_train_feed_ascend_graphmode():
    '''
    Feature: shard function for cell to enable parallel execution under Graph mode in Ascend
    Description: Test a shrunk version of ResNet50 with a alternative execution of shard and Graph
    Expectation: Run success
    '''
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    rt.set_memory(max_size="25GB")
    context.set_context(device_id=int(os.getenv('DEVICE_ID')))
    init()
    context.set_auto_parallel_context(gradients_mean=True, parallel_mode=ParallelMode.AUTO_PARALLEL,
                                      search_mode="sharding_propagation", device_num=8,
                                      dataset_strategy="data_parallel")
    np.random.seed(42)
    set_seed(42)
    train_feed(num_classes=65536)

def test_train_feed_gpu_graphmode():
    '''
    Feature: shard function for cell to enable parallel execution under Graph mode in GPU
    Description: Test a shrunk version of ResNet50 with a alternative execution of shard and Graph
    Expectation: Run success
    '''
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    context.set_context(device_id=int(os.getenv('DEVICE_ID')))
    init()
    context.set_auto_parallel_context(gradients_mean=True, parallel_mode=ParallelMode.AUTO_PARALLEL,
                                      search_mode="sharding_propagation", device_num=8,
                                      dataset_strategy="data_parallel")
    np.random.seed(1)
    set_seed(1)
    train_feed(num_classes=65536)
