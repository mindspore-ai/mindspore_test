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
import shutil
import glob
from mindspore import dtype as mstype
from mindspore import Tensor, ops, nn, jit

graph_save_path = './graph_save_path'


def setup_function():
    os.environ['MS_DEV_SAVE_GRAPHS'] = '1'
    os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = './graph_save_path'
    os.environ['MS_DEV_DUMP_IR_PASSES'] = '_validate'
    if os.path.exists(graph_save_path):
        shutil.rmtree(graph_save_path)


def teardown_function():
    os.unsetenv('MS_DEV_SAVE_GRAPHS')
    os.unsetenv('MS_DEV_SAVE_GRAPHS_PATH')
    os.unsetenv('MS_DEV_DUMP_IR_PASSES')
    shutil.rmtree(graph_save_path)


def test_jit_ast_decorator_on_function():
    """
    Feature: Use jit api to create a callable MindSpore graph.
    Description: Use the jit api as a function decorator.
    Expectation: Success to create a callable MindSpore graph.
    """

    class AddNet(nn.Cell):
        def __init__(self):
            super(AddNet, self).__init__()
            self.add = ops.Add()

        @jit
        def construct(self, x, y):
            out = self.add(x, y)
            return out

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3]], dtype=mstype.float32)
    net = AddNet()
    net(x, y)
    assert glob.glob(os.path.join(graph_save_path, '*_validate*.ir'))


def test_jit_ast_decorator_on_class():
    """
    Feature: Use jit api to create a callable MindSpore graph.
    Description: Use the jit api as a class decorator.
    Expectation: Success to create a callable MindSpore graph.
    """
    @jit
    class AddNet(nn.Cell):
        def __init__(self):
            super(AddNet, self).__init__()
            self.add = ops.Add()

        def construct(self, x, y):
            out = self.add(x, y)
            return out

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3]], dtype=mstype.float32)
    net = AddNet()
    net(x, y)
    assert glob.glob(os.path.join(graph_save_path, '*_validate*.ir'))


def test_jit_bytecode_decorator_on_function():
    """
    Feature: Use jit api to create a callable MindSpore graph.
    Description: Use the jit api as a function decorator.
    Expectation: Success to create a callable MindSpore graph.
    """

    class AddNet(nn.Cell):
        def __init__(self):
            super(AddNet, self).__init__()
            self.add = ops.Add()

        @jit(capture_mode="bytecode")
        def construct(self, x, y):
            out = self.add(x, y)
            return out

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3]], dtype=mstype.float32)
    net = AddNet()
    net(x, y)
    assert glob.glob(os.path.join(graph_save_path, '*_validate*.ir'))


def test_jit_bytecode_decorator_on_class():
    """
    Feature: Use jit api to create a callable MindSpore graph.
    Description: Use the jit api as a class decorator.
    Expectation: Success to create a callable MindSpore graph.
    """
    @jit(capture_mode="bytecode")
    class AddNet(nn.Cell):
        def __init__(self):
            super(AddNet, self).__init__()
            self.add = ops.Add()

        def construct(self, x, y):
            out = self.add(x, y)
            return out

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3]], dtype=mstype.float32)
    net = AddNet()
    net(x, y)
    assert glob.glob(os.path.join(graph_save_path, '*_validate*.ir'))


def test_jit_trace_decorator_on_function():
    """
    Feature: Use jit api to create a callable MindSpore graph.
    Description: Use the jit api as a function decorator.
    Expectation: Success to create a callable MindSpore graph.
    """

    class AddNet(nn.Cell):
        def __init__(self):
            super(AddNet, self).__init__()
            self.add = ops.Add()

        @jit(capture_mode="trace")
        def construct(self, x, y):
            out = self.add(x, y)
            return out

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3]], dtype=mstype.float32)
    net = AddNet()
    net(x, y)
    assert glob.glob(os.path.join(graph_save_path, '*_validate*.ir'))


def test_jit_trace_decorator_on_class():
    """
    Feature: Use jit api to create a callable MindSpore graph.
    Description: Use the jit api as a class decorator.
    Expectation: Success to create a callable MindSpore graph.
    """
    @jit(capture_mode="trace")
    class AddNet(nn.Cell):
        def __init__(self):
            super(AddNet, self).__init__()
            self.add = ops.Add()

        def construct(self, x, y):
            out = self.add(x, y)
            return out

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3]], dtype=mstype.float32)
    net = AddNet()
    net(x, y)
    assert glob.glob(os.path.join(graph_save_path, '*_validate*.ir'))


def test_jit_ast_function_for_construct():
    """
    Feature: Use jit api to create a callable MindSpore graph.
    Description: Use the jit api as a function.
    Expectation: Success to create a callable MindSpore graph.
    """
    class AddNet(nn.Cell):
        def __init__(self):
            super(AddNet, self).__init__()
            self.add = ops.Add()

        def construct(self, x, y):
            out = self.add(x, y)
            return out

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3]], dtype=mstype.float32)
    net = AddNet()
    net.construct = jit(net.construct)
    net(x, y)
    assert glob.glob(os.path.join(graph_save_path, '*_validate*.ir'))
    shutil.rmtree(graph_save_path)
    AddNet.construct = jit(AddNet.construct)
    net = AddNet()
    net(x, y)
    assert glob.glob(os.path.join(graph_save_path, '*_validate*.ir'))


def test_jit_bytecode_function_for_construct():
    """
    Feature: Use jit api to create a callable MindSpore graph.
    Description: Use the jit api as a function.
    Expectation: Success to create a callable MindSpore graph.
    """
    class AddNet(nn.Cell):
        def __init__(self):
            super(AddNet, self).__init__()
            self.add = ops.Add()

        def construct(self, x, y):
            out = self.add(x, y)
            return out

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3]], dtype=mstype.float32)
    a = Tensor([[0.5, 0.6], [1.2, 1.3]], dtype=mstype.float32)
    b = Tensor([[0.01, 0.3], [0.1, 0.2]], dtype=mstype.float32)
    net = AddNet()
    net.construct = jit(net.construct, capture_mode="bytecode")
    net(x, y)
    assert glob.glob(os.path.join(graph_save_path, '*_validate*.ir'))
    shutil.rmtree(graph_save_path)
    AddNet.construct = jit(AddNet.construct, capture_mode="bytecode")
    net = AddNet()
    # Change the input shapes to avoid caching
    net(a, b)
    assert glob.glob(os.path.join(graph_save_path, '*_validate*.ir'))


def test_jit_trace_function_for_construct():
    """
    Feature: Use jit api to create a callable MindSpore graph.
    Description: Use the jit api as a function.
    Expectation: Success to create a callable MindSpore graph.
    """
    class AddNet(nn.Cell):
        def __init__(self):
            super(AddNet, self).__init__()
            self.add = ops.Add()

        def construct(self, x, y):
            out = self.add(x, y)
            return out

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3]], dtype=mstype.float32)
    net = AddNet()
    net.construct = jit(net.construct, capture_mode="trace")
    net(x, y)
    assert glob.glob(os.path.join(graph_save_path, '*_validate*.ir'))
    shutil.rmtree(graph_save_path)
    AddNet.construct = jit(AddNet.construct, capture_mode="trace")
    net = AddNet()
    net(x, y)
    assert glob.glob(os.path.join(graph_save_path, '*_validate*.ir'))


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.add = ops.Add()

    def construct(self, x, y):
        out = self.add(x, y)
        return out


def test_jit_ast_function_for_cell_instance():
    """
    Feature: Use jit api to create a callable MindSpore graph.
    Description: Use the jit api as a function.
    Expectation: Success to create a callable MindSpore graph.
    """

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3]], dtype=mstype.float32)
    net = jit(Net())
    net(x, y)
    assert glob.glob(os.path.join(graph_save_path, '*_validate*.ir'))


def test_jit_bytecode_function_for_cell_instance():
    """
    Feature: Use jit api to create a callable MindSpore graph.
    Description: Use the jit api as a function.
    Expectation: Success to create a callable MindSpore graph.
    """

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3]], dtype=mstype.float32)
    net = jit(Net(), capture_mode="bytecode")
    net(x, y)
    assert glob.glob(os.path.join(graph_save_path, '*_validate*.ir'))


def test_jit_trace_function_for_cell_instance():
    """
    Feature: Use jit api to create a callable MindSpore graph.
    Description: Use the jit api as a function.
    Expectation: Success to create a callable MindSpore graph.
    """

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3]], dtype=mstype.float32)
    net = jit(Net(), capture_mode="trace")
    net(x, y)
    assert glob.glob(os.path.join(graph_save_path, '*_validate*.ir'))


def test_jit_ast_function_for_cell_class():
    """
    Feature: Use jit api to create a callable MindSpore graph.
    Description: Use the jit api as a function.
    Expectation: Success to create a callable MindSpore graph.
    """
    class AddNet(nn.Cell):
        def __init__(self):
            super(AddNet, self).__init__()
            self.add = ops.Add()

        def construct(self, x, y):
            out = self.add(x, y)
            return out

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3]], dtype=mstype.float32)
    new_add_net = jit(AddNet)
    net = new_add_net()
    net(x, y)
    assert glob.glob(os.path.join(graph_save_path, '*_validate*.ir'))


def test_jit_bytecode_function_for_cell_class():
    """
    Feature: Use jit api to create a callable MindSpore graph.
    Description: Use the jit api as a function.
    Expectation: Success to create a callable MindSpore graph.
    """

    class AddNet(nn.Cell):
        def __init__(self):
            super(AddNet, self).__init__()
            self.add = ops.Add()

        def construct(self, x, y):
            out = self.add(x, y)
            return out

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3]], dtype=mstype.float32)
    new_add_net = jit(Net, capture_mode="bytecode")
    net = new_add_net()
    net(x, y)
    assert glob.glob(os.path.join(graph_save_path, '*_validate*.ir'))


def test_jit_trace_function_for_cell_class():
    """
    Feature: Use jit api to create a callable MindSpore graph.
    Description: Use the jit api as a function.
    Expectation: Success to create a callable MindSpore graph.
    """

    class AddNet(nn.Cell):
        def __init__(self):
            super(AddNet, self).__init__()
            self.add = ops.Add()

        def construct(self, x, y):
            out = self.add(x, y)
            return out

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3]], dtype=mstype.float32)
    new_add_net = jit(Net, capture_mode="trace")
    net = new_add_net()
    net(x, y)
    assert glob.glob(os.path.join(graph_save_path, '*_validate*.ir'))


def test_jit_ast_function_for_cell_instance_twice():
    """
    Feature: Use jit api to create two callable MindSpore graph for a cell instance.
    Description: Use the jit api as a function.
    Expectation: Success to create two callable MindSpore graphs and compile once.
    """

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3]], dtype=mstype.float32)
    net = Net()
    jit_net1 = jit(net)
    jit_net1(x, y)
    jit_net2 = jit(net)
    jit_net2(x, y)
    glob_list = glob.glob(os.path.join(graph_save_path, '*_validate*.ir'))
    assert len(glob_list) == 1


def test_jit_ast_function_for_cell_instance_twice_and_set_graph_name():
    """
    Feature: Use jit api to create two callable MindSpore graph for a cell instance but reset graph name.
    Description: Use the jit api as a function.
    Expectation: Success to create two callable MindSpore graphs and compile twice.
    """

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3]], dtype=mstype.float32)
    net = Net()
    jit_net1 = jit(net)
    jit_net1(x, y)
    net._set_jit_graph_name('second_net')  # pylint: disable=protected-access
    jit_net2 = jit(net)
    jit_net2(x, y)
    glob_list = glob.glob(os.path.join(graph_save_path, '*_validate*.ir'))
    assert len(glob_list) == 2
