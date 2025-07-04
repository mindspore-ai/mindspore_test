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
import socket
import shutil

import mindspore.nn as nn
from mindspore import context
from mindspore import Parameter, Tensor, save_checkpoint
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train import CheckpointConfig
from mindspore.train._utils import get_parameter_redundancy, remove_param_redundancy
from tests.mark_utils import arg_mark


def is_port_free(port):
    """Check if the specified port is not occupied."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(('0.0.0.0', port))
        return True
    except socket.error as e:
        if e.errno == socket.errno.EADDRINUSE:
            return False
        return False
    finally:
        s.close()


def set_port():
    """Set hccl port."""
    for i in range(61000, 65400, 16):
        flag = True
        for j in range(i, i + 16):
            if not is_port_free(j):
                flag = False
                break
        if flag:
            os.environ["HCCL_IF_BASE_PORT"] = str(i)
            break


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
    'accu_grads.backbone.blocks.16.output.mapping.weight':
        ([4, 4], [-1, 0], [2560, 2560], 0, True, '4-11650191013956257822'),
    'accu_grads.backbone.blocks.16.layernorm1.gamma':
        ([4, 4], [-1], [2560], 0, True, ''),
    'accu_grads.backbone.blocks.16.attention.dense1.bias':
        ([4, 4], [0], [640], 0, True, ''),
}


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='allcards', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_remove_redundancy_1_1(mode):
    '''
    Feature: remove_redundancy save ckpt and load ckpt.
    Description: Saving and loading checkpoints with redundancy elimination.
    Expectation: success.
    '''
    for i in range(8):
        os.mkdir(f"device{i}_redundancy11")
    set_port()
    ret = os.system("msrun --worker_num=8 --local_worker_num=8 --join=True " \
                    "pytest -s remove_redundancy.py::test_remove_redundancy_save_True_load_True")
    assert ret == 0
    for i in range(8):
        shutil.rmtree(f"device{i}_redundancy11")


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='allcards', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_remove_redundancy_1_0(mode):
    '''
    Feature: save remove_redundancy ckpt and full load ckpt.
    Description: Redundant-free checkpoint saving and full checkpoint loading.
    Expectation: success.
    '''
    for i in range(8):
        os.mkdir(f"device{i}_redundancy10")
    set_port()
    ret = os.system("msrun --worker_num=8 --local_worker_num=8 --join=True " \
                    "pytest -s remove_redundancy.py::test_remove_redundancy_save_True_load_False")
    assert ret == 0
    for i in range(8):
        shutil.rmtree(f"device{i}_redundancy10")


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='allcards', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_remove_redundancy_0_0(mode):
    '''
    Feature: save ckpt and load ckpt.
    Description: Full checkpoint saving and full checkpoint loading.
    Expectation: success.
    '''
    for i in range(8):
        os.mkdir(f"device{i}_redundancy00")
    set_port()
    ret = os.system("msrun --worker_num=8 --local_worker_num=8 --join=True " \
                    "pytest -s remove_redundancy.py::test_remove_redundancy_save_False_load_False")
    assert ret == 0
    for i in range(8):
        shutil.rmtree(f"device{i}_redundancy00")


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_save_remove_redundancy_error(mode):
    '''
    Feature: Verify error reporting during redundant-free saving.
    Description: Verify error reporting during redundant-free saving.
    Expectation: success.
    '''
    with pytest.raises(ValueError):
        CheckpointConfig(remove_redundancy="string")


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='allcards', essential_mark='unessential')
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


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='allcards', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_remove_redundancy_1_1_dp(mode):
    '''
    Feature: remove_redundancy save ckpt and load ckpt in data parallel.
    Description: Saving and loading checkpoints with redundancy elimination.
    Expectation: success.
    '''
    for i in range(8):
        os.mkdir(f"device{i}_redundancy11dp")
    set_port()
    ret = os.system("msrun --worker_num=8 --local_worker_num=8 --join=True " \
                    "pytest -s remove_redundancy_dp.py::test_remove_redundancy_save_True_load_True_dp")
    assert ret == 0
    for i in range(8):
        shutil.rmtree(f"device{i}_redundancy11dp")



@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_remove_redundancy_algorithm(mode):
    """
    Feature: Verify the redundancy removal algorithm.
    Description: Verify that the redundancy removal algorithm is correct.
    Expectation: run success
    """
    param_redundancy_dict = get_parameter_redundancy(parameter_layout_dict, initial_rank=0)
    single_parameter = remove_param_redundancy(param_redundancy_dict)
    expect_dict = {0: {'accu_grads.backbone.blocks.16.attention.dense1.bias',
                       'accu_grads.backbone.blocks.16.attention.projection.weight',
                       'accu_grads.backbone.blocks.16.output.mapping.weight',
                       'accu_grads.backbone.blocks.16.layernorm1.gamma'},
                   4: {'accu_grads.backbone.embedding.word_embedding.embedding_table',
                       'accu_grads.backbone.blocks.16.attention.projection.weight',
                       'accu_grads.backbone.blocks.16.output.mapping.weight'},
                   1: {'accu_grads.backbone.blocks.16.attention.dense1.bias',
                       'accu_grads.backbone.blocks.16.attention.projection.weight',
                       'accu_grads.backbone.blocks.16.output.mapping.weight'},
                   5: {'accu_grads.backbone.embedding.word_embedding.embedding_table',
                       'accu_grads.backbone.blocks.16.attention.projection.weight',
                       'accu_grads.backbone.blocks.16.output.mapping.weight'},
                   2: {'accu_grads.backbone.blocks.16.attention.dense1.bias',
                       'accu_grads.backbone.blocks.16.attention.projection.weight',
                       'accu_grads.backbone.blocks.16.output.mapping.weight'},
                   6: {'accu_grads.backbone.embedding.word_embedding.embedding_table',
                       'accu_grads.backbone.blocks.16.attention.projection.weight',
                       'accu_grads.backbone.blocks.16.output.mapping.weight'},
                   3: {'accu_grads.backbone.blocks.16.attention.dense1.bias',
                       'accu_grads.backbone.blocks.16.attention.projection.weight',
                       'accu_grads.backbone.blocks.16.output.mapping.weight'},
                   7: {'accu_grads.backbone.embedding.word_embedding.embedding_table',
                       'accu_grads.backbone.blocks.16.attention.projection.weight',
                       'accu_grads.backbone.blocks.16.output.mapping.weight'},
                   8: {'accu_grads.backbone.blocks.16.attention.projection.weight',
                       'accu_grads.backbone.blocks.16.output.mapping.weight'},
                   12: {'accu_grads.backbone.blocks.16.attention.projection.weight',
                        'accu_grads.backbone.blocks.16.output.mapping.weight'},
                   9: {'accu_grads.backbone.blocks.16.attention.projection.weight',
                       'accu_grads.backbone.blocks.16.output.mapping.weight'},
                   13: {'accu_grads.backbone.blocks.16.attention.projection.weight',
                        'accu_grads.backbone.blocks.16.output.mapping.weight'},
                   10: {'accu_grads.backbone.blocks.16.attention.projection.weight',
                        'accu_grads.backbone.blocks.16.output.mapping.weight'},
                   14: {'accu_grads.backbone.blocks.16.attention.projection.weight',
                        'accu_grads.backbone.blocks.16.output.mapping.weight'},
                   11: {'accu_grads.backbone.blocks.16.attention.projection.weight',
                        'accu_grads.backbone.blocks.16.output.mapping.weight'},
                   15: {'accu_grads.backbone.blocks.16.attention.projection.weight',
                        'accu_grads.backbone.blocks.16.output.mapping.weight'}}
    assert single_parameter == expect_dict


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='allcards', essential_mark='essential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_no_init_parameters_without_load_param(mode):
    '''
    Feature: no_init_parameters.
    Description: no_init_parameters with init_parameters_data.
    Expectation: success.
    '''
    for i in range(8):
        os.mkdir(f"device{i}_no_init_parameters")
    set_port()
    ret = os.system("msrun --worker_num=8 --local_worker_num=8 --join=True " \
                    "pytest -s remove_redundancy.py::test_no_init_parameters")
    assert ret == 0
    for i in range(8):
        shutil.rmtree(f"device{i}_no_init_parameters")
