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
import os
import pytest
from mindspore import context
import shutil

import mindspore.nn as nn
from mindspore import Parameter, Tensor, save_checkpoint
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train import CheckpointConfig
from mindspore.train._utils import get_parameter_redundancy, remove_param_redundancy


class MyCell(nn.Cell):
    def __init__(self):
        super(MyCell, self).__init__()
        self.param = Parameter(Tensor([1, 2, 3]))

    def construct(self, x):
        return x + self.param


parameter_layout_dict = {
    'accu_grads.backbone.embedding.word_embedding.embedding_table':
        ([4, 4], [0, -1], [10000, 2560], 0, True, ''),
    'accu_grads.backbone.blocks.16.attention.projection.weight':
        ([4, 4], [0, -1], [640, 2560], 0, True, '4-11650191013956257822'),
    'accu_grads.backbone.blocks.16.attention.dense1.weight':
        ([4, 4], [0, -1], [640, 2560], 0, True, '4-11650191013956257822'),
    'accu_grads.backbone.blocks.16.attention.dense2.weight':
        ([4, 4], [0, -1], [640, 2560], 0, True, '4-11650191013956257822'),
    'accu_grads.backbone.blocks.16.attention.dense3.weight':
        ([4, 4], [0, -1], [640, 2560], 0, True, '4-11650191013956257822'),
    'accu_grads.backbone.blocks.16.output.mapping.weight':
        ([4, 4], [-1, 0], [2560, 2560], 0, True, '4-11650191013956257822'),
    'accu_grads.backbone.blocks.16.layernorm1.gamma':
        ([4, 4], [-1], [2560], 0, True, ''),
    'accu_grads.backbone.blocks.16.layernorm1.beta':
        ([4, 4], [-1], [2560], 0, True, ''),
    'accu_grads.backbone.blocks.16.layernorm2.gamma':
        ([4, 4], [-1], [2560], 0, True, ''),
    'accu_grads.backbone.blocks.16.attention.dense1.bias':
        ([4, 4], [0], [640], 0, True, ''),
}


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_single
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_remove_redundancy_1_1(mode):
    '''
    Feature: remove_redundancy save ckpt and load ckpt.
    Description: Saving and loading checkpoints with redundancy elimination.
    Expectation: success.
    '''
    for i in range(8):
        os.mkdir(f"device{i}_redundancy11")
    ret = os.system("msrun --worker_num=8 --local_worker_num=8 --join=True " \
                    "pytest -s remove_redundancy.py::test_remove_redundancy_save_True_load_True")
    assert ret == 0
    for i in range(8):
        shutil.rmtree(f"device{i}_redundancy11")


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_single
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_remove_redundancy_1_0(mode):
    '''
    Feature: save remove_redundancy ckpt and full load ckpt.
    Description: Redundant-free checkpoint saving and full checkpoint loading.
    Expectation: success.
    '''
    for i in range(8):
        os.mkdir(f"device{i}_redundancy10")
    ret = os.system("msrun --worker_num=8 --local_worker_num=8 --join=True " \
                    "pytest -s remove_redundancy.py::test_remove_redundancy_save_True_load_False")
    assert ret == 0
    for i in range(8):
        shutil.rmtree(f"device{i}_redundancy10")


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_single
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_remove_redundancy_0_0(mode):
    '''
    Feature: save ckpt and load ckpt.
    Description: Full checkpoint saving and full checkpoint loading.
    Expectation: success.
    '''
    for i in range(8):
        os.mkdir(f"device{i}_redundancy00")
    ret = os.system("msrun --worker_num=8 --local_worker_num=8 --join=True " \
                    "pytest -s remove_redundancy.py::test_remove_redundancy_save_False_load_False")
    assert ret == 0
    for i in range(8):
        shutil.rmtree(f"device{i}_redundancy00")


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_single
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_save_remove_redundancy_error(mode):
    '''
    Feature: Verify error reporting during redundant-free saving.
    Description: Verify error reporting during redundant-free saving.
    Expectation: success.
    '''
    with pytest.raises(ValueError):
        CheckpointConfig(remove_redundancy="string")


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_single
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_load_remove_redundancy_error(mode):
    '''
    Feature: Verify error reporting during redundant-free loading.
    Description: Verify error reporting during redundant-free loading.
    Expectation: success.
    '''
    net = MyCell()
    save_checkpoint(net, "./redundancy_error.ckpt")
    param_dict = load_checkpoint("./redundancy_error.ckpt")

    with pytest.raises(ValueError):
        load_checkpoint("./redundancy_error.ckpt", remove_redundancy="string")
    with pytest.raises(ValueError):
        load_param_into_net(net, param_dict, remove_redundancy="string")


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_single
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_remove_redundancy_1_1_dp(mode):
    '''
    Feature: remove_redundancy save ckpt and load ckpt in data parallel.
    Description: Saving and loading checkpoints with redundancy elimination.
    Expectation: success.
    '''
    for i in range(8):
        os.mkdir(f"device{i}_redundancy11dp")
    ret = os.system("msrun --worker_num=8 --local_worker_num=8 --join=True " \
                    "pytest -s remove_redundancy_dp.py::test_remove_redundancy_save_True_load_True_dp")
    assert ret == 0
    for i in range(8):
        shutil.rmtree(f"device{i}_redundancy11dp")


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_single
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_remove_redundancy_algorithm(mode):
    """
    Feature: Verify the redundancy removal algorithm.
    Description: Verify that the redundancy removal algorithm is correct.
    Expectation: run success
    """
    param_redundancy_dict = get_parameter_redundancy(parameter_layout_dict)
    single_parameter = remove_param_redundancy(param_redundancy_dict)
    expect_dict = {
        0: {'accu_grads.backbone.blocks.16.output.mapping.weight', 'accu_grads.backbone.blocks.16.layernorm1.gamma',
            'accu_grads.backbone.blocks.16.attention.dense3.weight',
            'accu_grads.backbone.blocks.16.attention.dense1.weight',
            'accu_grads.backbone.embedding.word_embedding.embedding_table',
            'accu_grads.backbone.blocks.16.attention.dense2.weight',
            'accu_grads.backbone.blocks.16.attention.projection.weight'},
        4: {'accu_grads.backbone.blocks.16.attention.dense1.bias',
            'accu_grads.backbone.blocks.16.output.mapping.weight',
            'accu_grads.backbone.blocks.16.attention.dense1.weight',
            'accu_grads.backbone.blocks.16.attention.dense3.weight',
            'accu_grads.backbone.blocks.16.attention.dense2.weight',
            'accu_grads.backbone.blocks.16.attention.projection.weight'},
        1: {'accu_grads.backbone.blocks.16.output.mapping.weight', 'accu_grads.backbone.blocks.16.layernorm1.beta',
            'accu_grads.backbone.blocks.16.attention.dense3.weight',
            'accu_grads.backbone.blocks.16.attention.dense1.weight',
            'accu_grads.backbone.embedding.word_embedding.embedding_table',
            'accu_grads.backbone.blocks.16.attention.dense2.weight',
            'accu_grads.backbone.blocks.16.attention.projection.weight'},
        5: {'accu_grads.backbone.blocks.16.attention.dense1.bias',
            'accu_grads.backbone.blocks.16.output.mapping.weight',
            'accu_grads.backbone.blocks.16.attention.dense1.weight',
            'accu_grads.backbone.blocks.16.attention.dense3.weight',
            'accu_grads.backbone.blocks.16.attention.dense2.weight',
            'accu_grads.backbone.blocks.16.attention.projection.weight'},
        2: {'accu_grads.backbone.blocks.16.output.mapping.weight',
            'accu_grads.backbone.blocks.16.attention.dense3.weight',
            'accu_grads.backbone.blocks.16.attention.dense1.weight',
            'accu_grads.backbone.embedding.word_embedding.embedding_table',
            'accu_grads.backbone.blocks.16.layernorm2.gamma', 'accu_grads.backbone.blocks.16.attention.dense2.weight',
            'accu_grads.backbone.blocks.16.attention.projection.weight'},
        6: {'accu_grads.backbone.blocks.16.attention.dense1.bias',
            'accu_grads.backbone.blocks.16.output.mapping.weight',
            'accu_grads.backbone.blocks.16.attention.dense1.weight',
            'accu_grads.backbone.blocks.16.attention.dense3.weight',
            'accu_grads.backbone.blocks.16.attention.dense2.weight',
            'accu_grads.backbone.blocks.16.attention.projection.weight'},
        3: {'accu_grads.backbone.blocks.16.output.mapping.weight',
            'accu_grads.backbone.blocks.16.attention.dense3.weight',
            'accu_grads.backbone.blocks.16.attention.dense1.weight',
            'accu_grads.backbone.embedding.word_embedding.embedding_table',
            'accu_grads.backbone.blocks.16.attention.dense2.weight',
            'accu_grads.backbone.blocks.16.attention.projection.weight'},
        7: {'accu_grads.backbone.blocks.16.attention.dense1.bias',
            'accu_grads.backbone.blocks.16.output.mapping.weight',
            'accu_grads.backbone.blocks.16.attention.dense1.weight',
            'accu_grads.backbone.blocks.16.attention.dense3.weight',
            'accu_grads.backbone.blocks.16.attention.dense2.weight',
            'accu_grads.backbone.blocks.16.attention.projection.weight'},
        8: {'accu_grads.backbone.blocks.16.output.mapping.weight',
            'accu_grads.backbone.blocks.16.attention.dense1.weight',
            'accu_grads.backbone.blocks.16.attention.dense3.weight',
            'accu_grads.backbone.blocks.16.attention.dense2.weight',
            'accu_grads.backbone.blocks.16.attention.projection.weight'},
        12: {'accu_grads.backbone.blocks.16.output.mapping.weight',
             'accu_grads.backbone.blocks.16.attention.dense1.weight',
             'accu_grads.backbone.blocks.16.attention.dense3.weight',
             'accu_grads.backbone.blocks.16.attention.dense2.weight',
             'accu_grads.backbone.blocks.16.attention.projection.weight'},
        9: {'accu_grads.backbone.blocks.16.output.mapping.weight',
            'accu_grads.backbone.blocks.16.attention.dense1.weight',
            'accu_grads.backbone.blocks.16.attention.dense3.weight',
            'accu_grads.backbone.blocks.16.attention.dense2.weight',
            'accu_grads.backbone.blocks.16.attention.projection.weight'},
        13: {'accu_grads.backbone.blocks.16.output.mapping.weight',
             'accu_grads.backbone.blocks.16.attention.dense1.weight',
             'accu_grads.backbone.blocks.16.attention.dense3.weight',
             'accu_grads.backbone.blocks.16.attention.dense2.weight',
             'accu_grads.backbone.blocks.16.attention.projection.weight'},
        10: {'accu_grads.backbone.blocks.16.output.mapping.weight',
             'accu_grads.backbone.blocks.16.attention.dense1.weight',
             'accu_grads.backbone.blocks.16.attention.dense3.weight',
             'accu_grads.backbone.blocks.16.attention.dense2.weight',
             'accu_grads.backbone.blocks.16.attention.projection.weight'},
        14: {'accu_grads.backbone.blocks.16.output.mapping.weight',
             'accu_grads.backbone.blocks.16.attention.dense1.weight',
             'accu_grads.backbone.blocks.16.attention.dense3.weight',
             'accu_grads.backbone.blocks.16.attention.dense2.weight',
             'accu_grads.backbone.blocks.16.attention.projection.weight'},
        11: {'accu_grads.backbone.blocks.16.output.mapping.weight',
             'accu_grads.backbone.blocks.16.attention.dense1.weight',
             'accu_grads.backbone.blocks.16.attention.dense3.weight',
             'accu_grads.backbone.blocks.16.attention.dense2.weight',
             'accu_grads.backbone.blocks.16.attention.projection.weight'},
        15: {'accu_grads.backbone.blocks.16.output.mapping.weight',
             'accu_grads.backbone.blocks.16.attention.dense1.weight',
             'accu_grads.backbone.blocks.16.attention.dense3.weight',
             'accu_grads.backbone.blocks.16.attention.dense2.weight',
             'accu_grads.backbone.blocks.16.attention.projection.weight'}}
    assert single_parameter == expect_dict
