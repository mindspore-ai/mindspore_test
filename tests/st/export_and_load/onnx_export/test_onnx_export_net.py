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
from io import BytesIO
import numpy as np
from tests.mark_utils import arg_mark

import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from mindspore.common.tensor import Tensor
from mindspore.common.initializer import TruncatedNormal
from mindspore.ops import operations as P


def weight_variable():
    return TruncatedNormal(0.02)


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="valid")


def fc_with_initialize(input_channels, out_channels):
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)


class LeNet5(nn.Cell):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.batch_size = 32
        self.conv1 = conv(1, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, 10)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reshape = P.Reshape()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.reshape(x, (self.batch_size, -1))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def encrypt_func(model_stream, key):
    plain_data = BytesIO()
    plain_data.write(model_stream)
    return plain_data.getvalue()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_export_lenet_onnx_with_modify_input():
    """
    Feature: Export encrypted LeNet to ONNX
    Description: Export encrypted LeNet to ONNX for modify input/output name
    Expectation: save successfully
    """
    network1 = LeNet5()

    file_name1 = "lenet_modify_input.onnx"
    input_tensor = Tensor(np.ones([32, 1, 32, 32]).astype(np.float32) * 0.01)
    ms.onnx.export(network1, input_tensor, file_name=file_name1, input_names=["input_x"])
    assert os.path.exists(file_name1)
    os.remove(file_name1)

    network2 = LeNet5()
    file_name2 = "lenet_modify_output.onnx"
    input_tensor = Tensor(np.ones([32, 1, 32, 32]).astype(np.float32) * 0.01)
    ms.onnx.export(network2, input_tensor, file_name=file_name2, output_names=["output_x"])
    assert os.path.exists(file_name2)
    os.remove(file_name2)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_export_lenet_onnx_with_encryption():
    """
    Feature: Export encrypted LeNet to ONNX
    Description: Test export API to save network with encryption into ONNX
    Expectation: save successfully
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    network = LeNet5()
    network.set_train()
    file_name = "lenet_preprocess.onnx"

    input_tensor = Tensor(np.ones([32, 1, 32, 32]).astype(np.float32) * 0.01)
    ms.export(network, input_tensor, file_name=file_name, file_format='ONNX',
              enc_key=b'123456789', enc_mode=encrypt_func)
    assert os.path.exists(file_name)
    os.remove(file_name)
