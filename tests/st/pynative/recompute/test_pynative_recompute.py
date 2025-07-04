# Copyright 2023 Huawei Technologies Co., Ltd
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

import random
import numpy as np

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as vision
import mindspore.nn as nn
import mindspore.ops.functional as F

from mindspore import Tensor, Parameter
from mindspore import context
from mindspore import ParameterTuple
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.ops import composite as CP
from mindspore.nn.optim.momentum import Momentum
from mindspore.nn.wrap.cell_wrapper import WithLossCell
from mindspore import ops
from mindspore import recompute
from tests.mark_utils import arg_mark
random.seed(1)
np.random.seed(1)
ds.config.set_seed(1)

grad_by_list = CP.GradOperation(get_by_list=True)


def weight_variable_0(shape):
    zeros = np.zeros(shape).astype(np.float32)
    return Tensor(zeros)


def weight_variable_1(shape):
    ones = np.ones(shape).astype(np.float32)
    return Tensor(ones)


def conv3x3(in_channels, out_channels, stride=1, padding=0):
    """3x3 convolution """
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=3, stride=stride, padding=padding, weight_init='XavierUniform',
                     has_bias=False, pad_mode="same")


def conv1x1(in_channels, out_channels, stride=1, padding=0):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=1, stride=stride, padding=padding, weight_init='XavierUniform',
                     has_bias=False, pad_mode="same")


def conv7x7(in_channels, out_channels, stride=1, padding=0):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=7, stride=stride, padding=padding, weight_init='XavierUniform',
                     has_bias=False, pad_mode="same")


def bn_with_initialize(out_channels):
    shape = (out_channels)
    mean = weight_variable_0(shape)
    var = weight_variable_1(shape)
    beta = weight_variable_0(shape)
    bn = nn.BatchNorm2d(out_channels, momentum=0.99, eps=0.00001, gamma_init='Uniform',
                        beta_init=beta, moving_mean_init=mean, moving_var_init=var)
    return bn


def bn_with_initialize_last(out_channels):
    shape = (out_channels)
    mean = weight_variable_0(shape)
    var = weight_variable_1(shape)
    beta = weight_variable_0(shape)
    bn = nn.BatchNorm2d(out_channels, momentum=0.99, eps=0.00001, gamma_init='Uniform',
                        beta_init=beta, moving_mean_init=mean, moving_var_init=var)
    return bn


def fc_with_initialize(input_channels, out_channels):
    return nn.Dense(input_channels, out_channels, weight_init='XavierUniform', bias_init='Uniform')


class ResidualBlock(nn.Cell):
    expansion = 4

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1):
        super(ResidualBlock, self).__init__()

        out_chls = out_channels // self.expansion
        self.conv1 = conv1x1(in_channels, out_chls, stride=stride, padding=0)
        self.bn1 = bn_with_initialize(out_chls)

        self.conv2 = conv3x3(out_chls, out_chls, stride=1, padding=0)
        self.bn2 = bn_with_initialize(out_chls)

        self.conv3 = conv1x1(out_chls, out_channels, stride=1, padding=0)
        self.bn3 = bn_with_initialize_last(out_channels)

        self.relu = P.ReLU()
        self.add = P.Add()

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.add(out, identity)
        out = self.relu(out)

        return out


class ResidualBlockWithDown(nn.Cell):
    expansion = 4

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 down_sample=False):
        super(ResidualBlockWithDown, self).__init__()

        out_chls = out_channels // self.expansion
        self.conv1 = conv1x1(in_channels, out_chls, stride=stride, padding=0)
        self.bn1 = bn_with_initialize(out_chls)

        self.conv2 = conv3x3(out_chls, out_chls, stride=1, padding=0)
        self.bn2 = bn_with_initialize(out_chls)

        self.conv3 = conv1x1(out_chls, out_channels, stride=1, padding=0)
        self.bn3 = bn_with_initialize_last(out_channels)

        self.relu = P.ReLU()
        self.downsample = down_sample

        self.conv_down_sample = conv1x1(in_channels, out_channels, stride=stride, padding=0)
        self.bn_down_sample = bn_with_initialize(out_channels)
        self.add = P.Add()

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.conv_down_sample(identity)
        identity = self.bn_down_sample(identity)

        out = self.add(out, identity)
        out = self.relu(out)

        return out


class MakeLayer0(nn.Cell):

    def __init__(self, block, in_channels, out_channels, stride):
        super(MakeLayer0, self).__init__()
        self.a = ResidualBlockWithDown(in_channels, out_channels, stride=1, down_sample=True)
        self.b = block(out_channels, out_channels, stride=stride)
        self.c = block(out_channels, out_channels, stride=1)

    def construct(self, x):
        x = self.a(x)
        x = self.b(x)
        x = self.c(x)

        return x


class MakeLayer1(nn.Cell):

    def __init__(self, block, in_channels, out_channels, stride):
        super(MakeLayer1, self).__init__()
        self.a = ResidualBlockWithDown(in_channels, out_channels, stride=stride, down_sample=True)
        self.b = block(out_channels, out_channels, stride=1)
        self.c = block(out_channels, out_channels, stride=1)
        self.d = block(out_channels, out_channels, stride=1)

    def construct(self, x):
        x = self.a(x)
        x = self.b(x)
        x = self.c(x)
        x = self.d(x)

        return x


class MakeLayer2(nn.Cell):

    def __init__(self, block, in_channels, out_channels, stride):
        super(MakeLayer2, self).__init__()
        self.a = ResidualBlockWithDown(in_channels, out_channels, stride=stride, down_sample=True)
        self.b = block(out_channels, out_channels, stride=1)
        self.c = block(out_channels, out_channels, stride=1)
        self.d = block(out_channels, out_channels, stride=1)
        self.e = block(out_channels, out_channels, stride=1)
        self.f = block(out_channels, out_channels, stride=1)

    def construct(self, x):
        x = self.a(x)
        x = self.b(x)
        x = self.c(x)
        x = self.d(x)
        x = self.e(x)
        x = self.f(x)

        return x


class MakeLayer3(nn.Cell):

    def __init__(self, block, in_channels, out_channels, stride):
        super(MakeLayer3, self).__init__()
        self.a = ResidualBlockWithDown(in_channels, out_channels, stride=stride, down_sample=True)
        self.b = block(out_channels, out_channels, stride=1)
        self.c = block(out_channels, out_channels, stride=1)

    def construct(self, x):
        x = self.a(x)
        x = self.b(x)
        x = self.c(x)

        return x


def cell_hook_function(cell_id, grad_input, grad_output):
    print("cell id:", cell_id)


class ResNet(nn.Cell):

    def __init__(self, block, num_classes=100, batch_size=32):
        super(ResNet, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.conv1 = conv7x7(3, 64, stride=2, padding=0)

        self.bn1 = bn_with_initialize(64)
        self.relu = P.ReLU()
        self.maxpool = P.MaxPoolWithArgmax(kernel_size=3, strides=2, pad_mode="SAME")

        self.layer1 = MakeLayer0(block, in_channels=64, out_channels=256, stride=1)
        self.layer1.recompute()
        self.layer1.register_backward_hook(cell_hook_function)
        self.layer2 = MakeLayer1(block, in_channels=256, out_channels=512, stride=2)
        self.layer3 = MakeLayer2(block, in_channels=512, out_channels=1024, stride=2)
        self.layer3.recompute()
        self.layer3.to_float(mstype.float16)
        self.layer4 = MakeLayer3(block, in_channels=1024, out_channels=2048, stride=2)

        self.pool = P.ReduceMean(keep_dims=True)
        self.squeeze = P.Squeeze(axis=(2, 3))
        self.fc = fc_with_initialize(512 * block.expansion, num_classes)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)[0]

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = ops.cast(x, mstype.float32)
        x = self.layer4(x)

        x = self.pool(x, (2, 3))
        x = self.squeeze(x)
        x = self.fc(x)
        return x


def resnet50(batch_size, num_classes):
    return ResNet(ResidualBlock, num_classes, batch_size)


def create_dataset(repeat_num=1, training=True, batch_size=32):
    data_home = "/home/workspace/mindspore_dataset"
    data_dir = data_home + "/cifar-10-batches-bin"
    if not training:
        data_dir = data_home + "/cifar-10-verify-bin"
    data_set = ds.Cifar10Dataset(data_dir)

    resize_height = 224
    resize_width = 224
    rescale = 1.0 / 255.0
    shift = 0.0

    # define map operations
    random_crop_op = vision.RandomCrop((32, 32), (4, 4, 4, 4))  # padding_mode default CONSTANT
    random_horizontal_op = vision.RandomHorizontalFlip()
    # interpolation default BILINEAR
    resize_op = vision.Resize((resize_height, resize_width))
    rescale_op = vision.Rescale(rescale, shift)
    normalize_op = vision.Normalize((0.4465, 0.4822, 0.4914), (0.2010, 0.1994, 0.2023))
    changeswap_op = vision.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    c_trans = []
    if training:
        c_trans = [random_crop_op, random_horizontal_op]
    c_trans += [resize_op, rescale_op, normalize_op,
                changeswap_op]

    # apply map operations on images
    data_set = data_set.map(operations=type_cast_op, input_columns="label")
    data_set = data_set.map(operations=c_trans, input_columns="image")

    # apply shuffle operations
    data_set = data_set.shuffle(buffer_size=1000)

    # apply batch operations
    data_set = data_set.batch(batch_size=batch_size, drop_remainder=True)

    # apply repeat operations
    data_set = data_set.repeat(repeat_num)

    return data_set


class CrossEntropyLoss(nn.Cell):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.cross_entropy = P.SoftmaxCrossEntropyWithLogits()
        self.mean = P.ReduceMean()
        self.one_hot = P.OneHot()
        self.one = Tensor(1.0, mstype.float32)
        self.zero = Tensor(0.0, mstype.float32)

    def construct(self, logits, label):
        label = self.one_hot(label, F.shape(logits)[1], self.one, self.zero)
        loss = self.cross_entropy(logits, label)[0]
        loss = self.mean(loss, (-1,))
        return loss


class GradWrap(Cell):
    """ GradWrap definition """

    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network
        self.weights = ParameterTuple(network.trainable_params())

    def construct(self, x, label):
        weights = self.weights
        return grad_by_list(self.network, weights)(x, label)


@arg_mark(plat_marks=['platform_gpu'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_resnet50_recompute_with_hook_and_mixed_precision():
    """
    Feature: Recompute with block, and set mix precision and backward hook
    Description: Each block is set recompute by the cell recompute api.
    Expectation: Run successfully and the memory usage is reduced.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU", max_device_memory="7GB")

    batch_size = 32
    num_classes = 10
    net = resnet50(batch_size, num_classes)
    criterion = CrossEntropyLoss()
    optimizer = Momentum(learning_rate=0.01, momentum=0.9,
                         params=filter(lambda x: x.requires_grad, net.get_parameters()))

    net_with_criterion = WithLossCell(net, criterion)
    net_with_criterion.set_grad()
    train_network = GradWrap(net_with_criterion)
    train_network.set_train()

    step = 0
    max_step = 20
    data_set = create_dataset(repeat_num=1, training=True, batch_size=batch_size)
    for element in data_set.create_dict_iterator(num_epochs=1):
        step = step + 1
        if step > max_step:
            break
        input_data = element["image"]
        input_label = element["label"]
        loss_output = net_with_criterion(input_data, input_label)
        grads = train_network(input_data, input_label)
        optimizer(grads)
        print("======step: ", step, " loss: ", loss_output.asnumpy())


class Block(Cell):
    def __init__(self):
        super(Block, self).__init__()
        self.transpose1 = P.Transpose()
        self.transpose2 = P.Transpose()
        self.transpose3 = P.Transpose()
        self.transpose4 = P.Transpose()
        self.real_div1 = P.RealDiv()
        self.real_div2 = P.RealDiv()
        self.batch_matmul1 = P.BatchMatMul()
        self.batch_matmul2 = P.BatchMatMul()
        self.add = P.Add()
        self.softmax = P.Softmax(-1)
        self.dropout = P.Dropout(0.9)
        self.expand_dims = P.ExpandDims()
        self.sub = P.Sub()
        self.mul = P.Mul()
        self.y = Parameter(Tensor(np.ones((8, 128, 128)).astype(np.float32)))

    def construct(self, x):
        transpose1 = self.transpose1(x, (0, 2, 1, 3))
        real_div1 = self.real_div1(transpose1, Tensor(2.37891))
        transpose2 = self.transpose2(x, (0, 2, 3, 1))
        real_div2 = self.real_div2(transpose2, Tensor(2.37891))
        batch_matmul1 = self.batch_matmul1(real_div1, real_div2)
        expand_dims = self.expand_dims(self.y, 1)
        sub = self.sub(Tensor([1.0]), expand_dims)
        mul = self.mul(sub, Tensor([-0.0001]))
        add = self.add(mul, batch_matmul1)
        soft_max = self.softmax(add)
        dropout = self.dropout(soft_max)
        transpose3 = self.transpose3(x, (0, 2, 1, 3))
        batch_matmul2 = self.batch_matmul2(dropout[0], transpose3)
        transpose4 = self.transpose4(batch_matmul2, (0, 2, 1, 3))
        return transpose4


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_net_normal_recompute():
    """
    Feature: Recompute with normal block
    Description: Each block is set recompute by the cell recompute api.
    Expectation: Run successfully and the memory usage is reduced.
    """

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.blocks = nn.CellList()
            for _ in range(3):
                b = OuterBlock()
                b.recompute()
                self.blocks.append(b)

        def construct(self, x):
            out = x
            for i in range(3):
                out = self.blocks[i](out)
            return out

    x = Tensor(np.ones((8, 128, 16, 32)).astype(np.float32))
    net = Net()
    grad_net = ops.GradOperation()(net)
    grad_net(x)


class OuterBlock(Cell):
    def __init__(self):
        super(OuterBlock, self).__init__()
        self.block = Block()

    def construct(self, x):
        return self.block(x)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_net_normal_recompute_function():
    """
    Feature: Recompute function with normal block
    Description: Each block is set recompute by the cell recompute api.
    Expectation: Run successfully and the memory usage is reduced.
    """

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.blocks = nn.CellList()
            for _ in range(3):
                b = OuterBlock()
                self.blocks.append(b)

        def construct(self, x):
            out = x
            for i in range(3):
                if i == 0:
                    out = recompute(self.blocks[i], out)
                else:
                    out = self.blocks[i](out)
            return out

    x = Tensor(np.ones((8, 128, 16, 32)).astype(np.float32))
    net = Net()
    grad_net = ops.GradOperation()(net)
    grad_net(x)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_net_normal_recompute_sequential_cell():
    """
    Feature: Recompute function with normal block
    Description: Each block is set recompute by the cell recompute api.
    Expectation: Run successfully and the memory usage is reduced.
    """

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.blocks = nn.SequentialCell(OuterBlock(), OuterBlock(), OuterBlock())
            self.blocks.recompute()

        def construct(self, x):
            out = self.blocks(x)
            return out

    x = Tensor(np.ones((8, 128, 16, 32)).astype(np.float32))
    net = Net()
    grad_net = ops.GradOperation()(net)
    grad_net(x)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_net_normal_recompute_not_tensor_input():
    """
    Feature: Recompute function with normal block
    Description: Each block is set recompute by the cell recompute api.
    Expectation: Run successfully and the memory usage is reduced.
    """

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.blocks = nn.SequentialCell(OuterBlock(), OuterBlock(), OuterBlock())
            self.blocks.recompute()

        def construct(self, x, y, z):
            out = self.blocks(x)
            return out

    x = Tensor(np.ones((8, 128, 16, 32)).astype(np.float32))
    y = Tensor(np.ones((8, 128, 16, 32)).astype(np.float32))
    z = Tensor(np.ones((8, 128, 16, 32)).astype(np.float32))
    net = Net()
    grad_net = ops.GradOperation()(net)
    grad_net(x, None, (y, z))


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_net_recompute_not_tensor_input():
    """
    Feature: Recompute function with normal block
    Description: Each block is set recompute by the cell recompute api.
    Expectation: Run successfully and the memory usage is reduced.
    """

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.block = OuterBlock()
            self.block.recompute()

        def construct(self, x, y, z):
            out = self.block(x)
            return out

    x = Tensor(np.ones((8, 128, 16, 32)).astype(np.float32))
    y = Tensor(np.ones((8, 128, 16, 32)).astype(np.float32))
    z = Tensor(np.ones((8, 128, 16, 32)).astype(np.float32))
    net = Net()
    grad_net = ops.GradOperation()(net)
    grad_net(x, None, (y, z))
