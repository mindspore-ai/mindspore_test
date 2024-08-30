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
import time
import shutil
import pytest
import numpy as np
import mindspore
from mindspore import nn, Tensor, ops, context, jit, Model
from mindspore.nn import Cell
from mindspore.nn.optim.momentum import Momentum
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.common.api import _cell_graph_executor, _MindsporeFunctionExecutor
from mindspore.common.parameter import Parameter
from mindspore import dataset as ds
from tests.security_utils import security_off_wrap
from tests.code_trace_analyzer import CodeTraceAnalyzer
from tests.ut.python.debug.resnet import resnet50, DatasetResNet

context.set_context(mode=context.GRAPH_MODE)


@security_off_wrap
def test_lenet_code_trace():
    """
    Feature: Code Trace.
    Description: Test Lenet code trace.
    Expectation: success.
    """

    class LeNet5(nn.Cell):
        def __init__(self, num_class=10, num_channel=1):
            super(LeNet5, self).__init__()
            self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
            self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
            self.relu = nn.ReLU()
            self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Dense(16 * 5 * 5, 120)
            self.fc2 = nn.Dense(120, 84)
            self.fc3 = nn.Dense(84, num_class)

        def construct(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.max_pool2d(x)
            x = self.conv2(x)
            x = self.relu(x)
            x = self.max_pool2d(x)
            x = self.flatten(x)
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    save_graph_path = "test_lenet_code_trace"
    context.set_context(save_graphs=1, save_graphs_path=save_graph_path)
    net = LeNet5()
    input_tensor = Tensor(np.ones([1, 1, 32, 32]).astype(np.float32) * 0.01)
    _cell_graph_executor.compile(net, input_tensor)

    analyzer = CodeTraceAnalyzer(net, save_graph_path, "validate")
    accuracy = analyzer.analyze()
    if accuracy != 1.0:
        analyzer.report_analysis()
        raise ValueError("Code trace accuracy is not 1.0")

    shutil.rmtree(save_graph_path)


@security_off_wrap
def test_resnet50_code_trace():
    """
    Feature: Code Trace.
    Description: Test ResNet50 code trace.
    Expectation: success.
    """

    save_graph_path = "test_resnet50_code_trace"
    context.set_context(save_graphs=1, save_graphs_path=save_graph_path)
    predict = Tensor(np.ones([32, 3, 224, 224]), dtype=mindspore.float32)
    label = Tensor(np.ones([32]), dtype=mindspore.int32)
    dataset = DatasetResNet(predict, label, 2)

    net = resnet50()
    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.1, 0.9)
    model = Model(net, loss, opt)
    model.train(1, dataset, dataset_sink_mode=False)

    analyzer = CodeTraceAnalyzer(net, save_graph_path, "validate")
    accuracy = analyzer.analyze()
    if accuracy != 1.0:
        analyzer.report_analysis()
        raise ValueError("Code trace accuracy is not 1.0")

    shutil.rmtree(save_graph_path)


@security_off_wrap
def test_resnet50_node_fullname():
    """
    Feature: Code Trace.
    Description: Test ResNet50 code trace.
    Expectation: success.
    """

    save_graph_path = "test_resnet50_node_fullname"
    context.set_context(save_graphs=1, save_graphs_path=save_graph_path)
    predict = Tensor(np.ones([32, 3, 224, 224]), dtype=mindspore.float32)
    label = Tensor(np.ones([32]), dtype=mindspore.int32)
    dataset = DatasetResNet(predict, label, 2)

    net = resnet50()
    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.1, 0.9)
    model = Model(net, loss, opt)
    model.train(1, dataset, dataset_sink_mode=False)

    analyzer = CodeTraceAnalyzer(net, save_graph_path, "validate")
    res = analyzer.check_fullname("ApplyMomentum-op", 160)
    if not res:
        raise ValueError("fullname of ApplyMomentum is changed by ir print")

    shutil.rmtree(save_graph_path)


@security_off_wrap
def test_code_trace1():
    """
    Feature: Code Trace.
    Description: Test source code location.
    Expectation: success.
    """

    class Net(nn.Cell):
        def construct(self, x, y, z):
            res = ops.maximum(x, y)
            res = ops.maximum(res, z)
            return res

    save_graph_path = "test_code_trace1"
    context.set_context(save_graphs=1, save_graphs_path=save_graph_path)
    net = Net()
    x = Tensor([1], mindspore.float32)
    y = Tensor([2], mindspore.float32)
    z = Tensor([3], mindspore.float32)
    _cell_graph_executor.compile(net, x, y, z)

    analyzer = CodeTraceAnalyzer(net, save_graph_path, "validate")
    accuracy = analyzer.analyze()
    if accuracy != 1.0:
        analyzer.report_analysis()
        raise ValueError("Code trace accuracy is not 1.0")

    shutil.rmtree(save_graph_path)


@security_off_wrap
def test_code_trace2():
    """
    Feature: Code Trace.
    Description: Test source code location.
    Expectation: success.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.relu = nn.ReLU()

        def construct(self, x):
            x = self.relu(x)
            x = self.relu(x)
            return x

    save_graph_path = "test_code_trace2"
    context.set_context(save_graphs=True, save_graphs_path=save_graph_path)
    net = Net()
    x = Tensor([1], mindspore.float32)
    _cell_graph_executor.compile(net, x)

    analyzer = CodeTraceAnalyzer(net, save_graph_path, "validate")
    accuracy = analyzer.analyze()
    if accuracy != 1.0:
        analyzer.report_analysis()
        raise ValueError("Code trace accuracy is not 1.0")

    shutil.rmtree(save_graph_path)


@security_off_wrap
def test_code_trace3():
    """
    Feature: Code Trace.
    Description: Test source code location.
    Expectation: success.
    """

    @jit
    def func(x, y):
        x = x.sum(-1)
        y = y.sum()
        return x + y

    save_graph_path = "test_code_trace3"
    context.set_context(save_graphs=True, save_graphs_path=save_graph_path)
    _ms_function_executor = _MindsporeFunctionExecutor(
        func, int(time.time() * 1e9))
    x = Tensor(np.arange(10).reshape(10).astype(np.float32))
    y = Tensor(np.array([-1, 0, 1]).astype(np.float32))
    _ms_function_executor.compile("fn", x, y)

    analyzer = CodeTraceAnalyzer(func, save_graph_path, "validate")
    accuracy = analyzer.analyze()
    if accuracy != 1.0:
        analyzer.report_analysis()
        raise ValueError("Code trace accuracy is not 1.0")

    shutil.rmtree(save_graph_path)


@pytest.mark.skip(reason="'x = self.dense1(x)' not in ir")
@security_off_wrap
def test_code_trace4():
    """
    Feature: Code Trace.
    Description: Test source code location.
    Expectation: success.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.dense1 = nn.Dense(3, 4)
            self.dense2 = nn.Dense(4, 5)

        def construct(self, x):
            x = self.dense1(x)
            x = self.dense2(x)
            return x

    save_graph_path = "test_code_trace4"
    context.set_context(save_graphs=True, save_graphs_path=save_graph_path)
    net = Net()
    x = Tensor(np.array([[180, 234, 154], [244, 48, 247]]), mindspore.float32)
    _cell_graph_executor.compile(net, x)

    analyzer = CodeTraceAnalyzer(net, save_graph_path, "validate")
    accuracy = analyzer.analyze()
    if accuracy != 1.0:
        analyzer.report_analysis()
        raise ValueError("Code trace accuracy is not 1.0")

    shutil.rmtree(save_graph_path)


@security_off_wrap
def test_code_trace5():
    """
    Feature: Code Trace.
    Description: Test source code location.
    Expectation: success.
    """

    class Net(nn.Cell):
        def construct(self, x):
            x = x[0, 1]
            x = x[2, 3]
            return x

    save_graph_path = "test_code_trace5"
    context.set_context(save_graphs=True, save_graphs_path=save_graph_path)
    net = Net()
    x = Tensor(np.ones([1, 120, 1024, 640]), mindspore.float32)
    _cell_graph_executor.compile(net, x)

    analyzer = CodeTraceAnalyzer(net, save_graph_path, "validate")
    accuracy = analyzer.analyze()
    if accuracy != 1.0:
        analyzer.report_analysis()
        raise ValueError("Code trace accuracy is not 1.0")

    shutil.rmtree(save_graph_path)


@security_off_wrap
def test_code_trace_loop_stack_depth():
    """
    Feature: Code Trace.
    Description: Test stack depth.
    Expectation: success.
    """
    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.default = 4000

        def construct(self, x):
            output = x
            for i in range(self.default):
                if self.default is None:
                    output = output + i
            if output > self.default:
                return self.default
            return output

    with pytest.raises(RuntimeError):
        context.set_context(mode=context.GRAPH_MODE)
        net = Net()
        x = Tensor([-1])
        out = net(x)
        assert out == -1


class Net1(nn.Cell):
    def construct(self, x1, x2):
        out = x1 + x2
        return out


class Net2(nn.Cell):
    def construct(self, x3, x4):
        out = x3 + x4
        return out


class Net3(nn.Cell):
    def __init__(self):
        super(Net3, self).__init__()
        self.net1 = Net1()
        self.net2 = Net2()

    def construct(self, x1, x2, x3):
        out1 = self.net1(x1, x2)
        out2 = self.net2(out1, x3)
        return out2


def add(x, y):
    return x + y

def mul(x, y):
    return x * y


def div(x, y):
    return x / y


class Net4(nn.Cell):
    def construct(self, x3, x4):
        out = add(x3, x4)
        return out


class Net5(nn.Cell):
    def __init__(self):
        super(Net5, self).__init__()
        self.net1 = Net4()
        self.net2 = Net2()

    def construct(self, x1, x2, x3):
        out1 = self.net1(x1, x2)
        out2 = self.net2(out1, x3)
        return out2


class Net6(nn.Cell):
    def __init__(self):
        super(Net6, self).__init__()
        self.net1 = Net2()
        self.net2 = Net4()

    def construct(self, x1, x2, x3):
        out1 = self.net1(x1, x2)
        out2 = self.net2(out1, x3)
        return out2


class Net7(nn.Cell):
    def __init__(self):
        super(Net7, self).__init__()
        self.net = Net4()
    def construct(self, x, y):
        out = self.net(x, y)
        return out

class Net8(nn.Cell):
    def __init__(self):
        super(Net8, self).__init__()
        self.net1 = Net7()
        self.net2 = Net4()

    def construct(self, x1, x2, x3):
        out1 = self.net1(x1, x2)
        out2 = self.net2(out1, x3)
        return out2


def cal(x, y):
    mul_res = mul(x, y)
    mul_res2 = mul(mul_res, x)
    div_res = div(mul_res2, y)
    out = add(div_res, mul_res)
    return out


class Net9(nn.Cell):
    def construct(self, x3, x4):
        out = add(x3, x4)
        return out


class Net10(nn.Cell):
    def construct(self, x3, x4):
        mul_res = mul(x3, x4)
        mul_res2 = mul(mul_res, x3)
        div_res = div(mul_res2, x4)
        out = add(div_res, mul_res)
        return out


class Net11(nn.Cell):
    def __init__(self):
        super(Net11, self).__init__()
        self.net1 = Net9()
        self.net2 = Net10()

    def construct(self, x1, x2, x3):
        out1 = self.net1(x1, x2)
        out2 = self.net2(out1, x3)
        return out2


@security_off_wrap
def test_code_trace6():
    """
    Feature: Code Trace.
    Description: Test source code location.
    Expectation: success.
    """

    save_graph_path = "test_code_trace6"
    context.set_context(save_graphs=True, save_graphs_path=save_graph_path)
    net = Net3()
    x1 = Tensor([[1, 2], [3, 4]])
    x2 = Tensor([[1.0, 2.0], [3.0, 4.0]])
    _cell_graph_executor.compile(net, x1, x1, x2)

    analyzer = CodeTraceAnalyzer(net, save_graph_path, "validate")

    # return lines of sub cells will not be counted since those cells are inlined
    accuracy = analyzer.analyze(2)
    if accuracy != 1.0:
        analyzer.report_analysis()
        raise ValueError("Code trace accuracy is not 1.0")

    shutil.rmtree(save_graph_path)


@security_off_wrap
def test_code_trace7():
    """
    Feature: Code Trace.
    Description: Test source code location.
    Expectation: success.
    """

    save_graph_path = "test_code_trace7"
    context.set_context(save_graphs=True, save_graphs_path=save_graph_path)
    net = Net3()
    x1 = Tensor([[1, 2], [3, 4]])
    x2 = Tensor([[1, 2], [3, 4]])
    _cell_graph_executor.compile(net, x1, x1, x2)

    analyzer = CodeTraceAnalyzer(net, save_graph_path, "validate")

    # return lines of sub cells will not be counted since those cells are inlined
    accuracy = analyzer.analyze(2)
    if accuracy != 1.0:
        analyzer.report_analysis()
        raise ValueError("Code trace accuracy is not 1.0")

    shutil.rmtree(save_graph_path)


@security_off_wrap
def test_code_trace8():
    """
    Feature: Code Trace.
    Description: Test source code location.
    Expectation: success.
    """

    class Net_Inner(nn.Cell):
        def construct(self, x1, x2, x3):
            out1 = x1 + x2
            out2 = out1 + x3
            return out2

    save_graph_path = "test_code_trace8"
    context.set_context(save_graphs=True, save_graphs_path=save_graph_path)
    net = Net_Inner()
    x1 = Tensor([[1, 2], [3, 4]])
    x2 = Tensor([[1.0, 2.0], [3.0, 4.0]])
    _cell_graph_executor.compile(net, x1, x1, x2)

    analyzer = CodeTraceAnalyzer(net, save_graph_path, "validate")
    accuracy = analyzer.analyze()
    if accuracy != 1.0:
        analyzer.report_analysis()
        raise ValueError("Code trace accuracy is not 1.0")

    shutil.rmtree(save_graph_path)


@security_off_wrap
def test_code_trace9():
    """
    Feature: Code Trace.
    Description: Test source code location.
    Expectation: success.
    """

    save_graph_path = "test_code_trace9"
    context.set_context(save_graphs=True, save_graphs_path=save_graph_path)
    net = Net5()
    x1 = Tensor([[1, 2], [3, 4]])
    x2 = Tensor([[1.0, 2.0], [3.0, 4.0]])
    _cell_graph_executor.compile(net, x1, x1, x2)

    analyzer = CodeTraceAnalyzer(net, save_graph_path, "validate")

    # return lines of sub cells will not be counted since those cells are inlined
    accuracy = analyzer.analyze(2)
    if accuracy != 1.0:
        analyzer.report_analysis()
        raise ValueError("Code trace accuracy is not 1.0")

    shutil.rmtree(save_graph_path)


@security_off_wrap
def test_code_trace10():
    """
    Feature: Code Trace.
    Description: Test source code location.
    Expectation: success.
    """

    save_graph_path = "test_code_trace10"
    context.set_context(save_graphs=True, save_graphs_path=save_graph_path)
    net = Net6()
    x1 = Tensor([[1, 2], [3, 4]])
    x2 = Tensor([[1.0, 2.0], [3.0, 4.0]])
    _cell_graph_executor.compile(net, x1, x1, x2)

    analyzer = CodeTraceAnalyzer(net, save_graph_path, "validate")

    # return lines of sub cells will not be counted since those cells are inlined
    accuracy = analyzer.analyze(2)
    if accuracy != 1.0:
        analyzer.report_analysis()
        raise ValueError("Code trace accuracy is not 1.0")

    shutil.rmtree(save_graph_path)


@security_off_wrap
def test_code_trace11():
    """
    Feature: Code Trace.
    Description: Test source code location.
    Expectation: success.
    """

    save_graph_path = "test_code_trace11"
    context.set_context(save_graphs=True, save_graphs_path=save_graph_path)
    net = Net8()
    x1 = Tensor([[1, 2], [3, 4]])
    x2 = Tensor([[1.0, 2.0], [3.0, 4.0]])
    _cell_graph_executor.compile(net, x1, x1, x2)

    analyzer = CodeTraceAnalyzer(net, save_graph_path, "validate")

    # return lines of sub cells will not be counted since those cells are inlined
    accuracy = analyzer.analyze(3)
    if accuracy != 1.0:
        analyzer.report_analysis()
        raise ValueError("Code trace accuracy is not 1.0")

    shutil.rmtree(save_graph_path)


@security_off_wrap
def test_code_trace12():
    """
    Feature: Code Trace.
    Description: Test source code location.
    Expectation: success.
    """

    save_graph_path = "test_code_trace12"
    context.set_context(save_graphs=True, save_graphs_path=save_graph_path)
    net = Net11()
    x1 = Tensor([[1, 2], [3, 4]])
    x2 = Tensor([[1.0, 2.0], [3.0, 4.0]])
    _cell_graph_executor.compile(net, x1, x1, x2)

    analyzer = CodeTraceAnalyzer(net, save_graph_path, "validate")

    # return lines of sub cells will not be counted since those cells are inlined
    accuracy = analyzer.analyze(2)
    if accuracy != 1.0:
        analyzer.report_analysis()
        raise ValueError("Code trace accuracy is not 1.0")

    shutil.rmtree(save_graph_path)


class AddNet(nn.Cell):
    def __init__(self, weight_shape, dtype=mindspore.float32):
        super(AddNet, self).__init__()
        self.add = ops.Add()
        self.add_weight = Parameter(
            Tensor(np.full(weight_shape, 2.0), dtype=dtype), name="matmul_weight")

    def construct(self, x, y):
        out = x + y
        out = self.add(out, self.add_weight)
        out = out + out
        return out


def generator(size, dtype=mindspore.float32, label_dtype=mindspore.float32):
    for _ in range(2):
        inputs, label = Tensor(np.full(size[0], 0.5), dtype=dtype), Tensor(np.full(size[1], 0.5),
                                                                           dtype=label_dtype)
        yield inputs, label


@pytest.mark.skip(reason="temperory skip for now")
@security_off_wrap
def test_code_trace13():
    """
    Feature: Code Trace.
    Description: Test source code location.
    Expectation: success.
    """

    save_graph_path = "test_code_trace13"
    context.set_context(save_graphs=True, save_graphs_path=save_graph_path)
    net = AddNet(weight_shape=(1, 2), dtype=mindspore.float32)


    opt_fn = Momentum(learning_rate=0.01, momentum=0.9,
                      params=net.get_parameters())
    dataset = ds.GeneratorDataset(lambda: generator(size=((2, 2), (2, 2)), label_dtype=mindspore.float32),
                                  ["inputs", "label"])
    model = Model(net, optimizer=opt_fn)
    model.train(2, dataset, dataset_sink_mode=False)

    analyzer = CodeTraceAnalyzer(net, save_graph_path, "validate")

    # return lines of sub cells will not be counted since those cells are inlined
    accuracy = analyzer.analyze(1)
    if accuracy != 1.0:
        analyzer.report_analysis()
        raise ValueError("Code trace accuracy is not 1.0")

    shutil.rmtree(save_graph_path)
