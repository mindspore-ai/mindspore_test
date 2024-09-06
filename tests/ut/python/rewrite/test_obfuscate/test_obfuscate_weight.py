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
"""test obfuscate weight"""
import pytest

from mindspore import nn
from mindspore.common.initializer import TruncatedNormal
from mindspore import obfuscate_ckpt, load_obf_params_into_net
import mindspore.ops as ops


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="valid")


def fc_with_initialize(input_channels, out_channels):
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)


def weight_variable():
    return TruncatedNormal(0.05)


class SubNet(nn.Cell):
    """
    SubNet of lenet.
    """
    def __init__(self):
        super(SubNet, self).__init__()
        self.dense_op = fc_with_initialize(84, 10)

    def construct(self, x):
        return self.dense_op(x)


class LeNet5(nn.Cell):
    """
    Lenet network
    """
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = conv(1, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.fc1 = fc_with_initialize(16*5*5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.sub_net = SubNet()
        self.relu = ops.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.sub_net(x)
        return x


def test_abnormal_ckpt_files():
    """
    Feature: Test weight obfuscation.
    Description: Test abnormal ckpt_files.
    Expectation: Raise TypeError.
    """
    net = LeNet5()
    obf_target_modules = ['sub_net', 'dense_op']
    with pytest.raises(TypeError):
        obfuscate_ckpt(net, ckpt_files=1, target_modules=obf_target_modules, saved_path='./obf_files')


def test_abnormal_target_modules_1():
    """
    Feature: Test weight obfuscation.
    Description: Test abnormal target_modules.
    Expectation: Raise TypeError.
    """
    net = LeNet5()
    obf_target_modules = ['sub_net', 1]
    with pytest.raises(TypeError):
        obfuscate_ckpt(net, ckpt_files='./', target_modules=obf_target_modules, saved_path='./')


def test_abnormal_obf_configs_1():
    """
    Feature: Test weight obfuscation.
    Description: Test abnormal obf_config.
    Expectation: Raise TypeError.
    """
    net = LeNet5()
    obf_configs = {}
    obf_configs['obf_metadata_config'] = [{
        'name': [1,],
        'shape': [1,],
        'type': 'random',
        'save_metadata': True,
        'metadata_op': 'invert'
    }]
    obf_configs['weight_obf_config'] = [{
        'target': 'sub_net/dense_op',
        'weight_obf_ops': [{'name': 'mul', 'input_x': 'weight', 'input_y': 'obf_metadata'}]
    }]
    obf_configs['network_obf_config'] = [{
        'module': 'sub_net',
        'target': 'dense_op',
        'insert_new_input': [{'name': 'obf_metadata'}],
        'insert_ops': [{'name': 'mul', 'input_x': 'weight', 'input_y': 'obf_metadata'}]
    }]
    with pytest.raises(TypeError) as info:
        obfuscate_ckpt(net, ckpt_files='./', obf_config=obf_configs, saved_path='./')
    assert "obf_config[][]['name'] type should be str" in str(info)


def test_abnormal_obf_configs_2():
    """
    Feature: Test weight obfuscation.
    Description: Test abnormal obf_config.
    Expectation: Raise TypeError.
    """
    net = LeNet5()
    obf_configs = {}
    obf_configs['obf_metadata_config'] = [{
        'name': 'obf_metadata',
        'shape': [1,],
        'type': 'random',
        'save_metadata': True,
        'metadata_op': 'invert'
    }]
    obf_configs['weight_obf_config'] = [{
        'target': 'sub_net/dense_op',
        'layers': "",
        'weight_obf_ops': [{'name': 'mul', 'input_x': 'weight', 'input_y': 'obf_metadata'}]
    }]
    obf_configs['network_obf_config'] = [{
        'module': 'sub_net',
        'target': 'dense_op',
        'insert_new_input': [{'name': 'obf_metadata'}],
        'insert_ops': [{'name': 'mul', 'input_x': 'weight', 'input_y': 'obf_metadata'}]
    }]
    with pytest.raises(TypeError) as info:
        obfuscate_ckpt(net, ckpt_files='./', obf_config=obf_configs, saved_path='./')
    assert "obf_config[][]['layers'] type should be list" in str(info)


def test_abnormal_obf_configs_3():
    """
    Feature: Test weight obfuscation.
    Description: Test abnormal obf_config.
    Expectation: Raise TypeError.
    """
    net = LeNet5()
    obf_configs = {}
    obf_configs['obf_metadata_config'] = [{
        'name': 'obf_metadata',
        'shape': [1,],
        'type': 'random',
        'save_metadata': True,
        'metadata_op': 'invert'
    }]
    obf_configs['weight_obf_config'] = [{
        'target': 'sub_net/dense_op',
        'weight_obf_ops': [{'name': 'mul', 'input_x': 'weight', 'input_y': 'obf_metadata'}]
    }]
    obf_configs['network_obf_config'] = [{
        'module': [1,],
        'target': 'dense_op',
        'insert_new_input': [{'name': 'obf_metadata'}],
        'insert_ops': [{'name': 'mul', 'input_x': 'weight', 'input_y': 'obf_metadata'}]
    }]
    with pytest.raises(TypeError) as info:
        load_obf_params_into_net(net, obf_config=obf_configs)
    assert "obf_config[][]['module'] type should be str" in str(info)


def test_abnormal_obf_configs_4():
    """
    Feature: Test weight obfuscation.
    Description: Test abnormal obf_config.
    Expectation: Raise TypeError.
    """
    net = LeNet5()
    obf_configs = {}
    obf_configs['config'] = [{
        'name': 'obf_metadata',
        'shape': [1,],
        'type': 'random',
        'save_metadata': True,
        'metadata_op': 'invert'
    }]
    obf_configs['weight_obf_config'] = [{
        'target': 'sub_net/dense_op',
        'weight_obf_ops': [{'name': 'mul', 'input_x': 'weight', 'input_y': 'obf_metadata'}]
    }]
    obf_configs['network_obf_config'] = [{
        'module': 'sub_net',
        'target': 'dense_op',
        'insert_new_input': [{'name': 'obf_metadata'}],
        'insert_ops': [{'name': 'mul', 'input_x': 'weight', 'input_y': 'obf_metadata'}]
    }]
    with pytest.raises(TypeError) as info:
        obfuscate_ckpt(net, ckpt_files='./', obf_config=obf_configs, saved_path='./')
    assert "config_type must be str, and in" in str(info)
