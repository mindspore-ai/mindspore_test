import sys  
import pytest 
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, context, jit
import numpy as np
from tests.mark_utils import arg_mark

@pytest.fixture(autouse=True)  
def skip_if_python_version_too_high():  
    if sys.version_info >= (3, 11):  
        pytest.skip("Skipping tests on Python 3.11 and higher.") 

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cell_dict_getitem():
    """
    Feature: Support CellDict in graph mode.
    Description: Verify the result of CellDict getitem.
    Expectation: Return expectation expected result.
    """

    class CellDictNet(nn.Cell):
        def __init__(self):
            super(CellDictNet, self).__init__()
            self.cell_dict = nn.CellDict([['conv', nn.Conv2d(6, 16, 5, pad_mode='valid')],
                                          ['relu', nn.ReLU()],
                                          ['max_pool2d', nn.MaxPool2d(kernel_size=4, stride=4)]]
                                         )

        @jit(mode="PIJit")
        def construct(self, key, x):
            op = self.cell_dict[key]
            return op(x)

    context.set_context(mode=context.PYNATIVE_MODE)
    cell_dict = {'conv': nn.Conv2d(6, 16, 5, pad_mode='valid'), 'relu': nn.ReLU(),
                 'max_pool2d': nn.MaxPool2d(kernel_size=4, stride=4)}
    net = CellDictNet()
    x = Tensor(np.ones([1, 6, 16, 5]), ms.float32)
    for key, cell in cell_dict.items():
        expect_output = cell(x)
        output = net(key, x)
        assert np.allclose(output.shape, expect_output.shape)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cell_dict_contain():
    """
    Feature: Support CellDict in graph mode.
    Description: Verify the result of CellDict contain.
    Expectation: Return expectation expected result.
    """

    class CellDictNet(nn.Cell):
        def __init__(self):
            super(CellDictNet, self).__init__()
            self.cell_dict = nn.CellDict([['conv', nn.Conv2d(6, 16, 5, pad_mode='valid')],
                                          ['relu', nn.ReLU()],
                                          ['max_pool2d', nn.MaxPool2d(kernel_size=4, stride=4)]]
                                         )

        @jit(mode="PIJit")
        def construct(self, key1, key2):
            ret1 = key1 in self.cell_dict
            ret2 = key2 in self.cell_dict
            return ret1, ret2

    context.set_context(mode=context.PYNATIVE_MODE)
    net = CellDictNet()
    out1, out2 = net("conv", "relu1")
    expect_out1 = True
    expect_out2 = False
    assert out1 == expect_out1
    assert out2 == expect_out2


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cell_dict_get_keys():
    """
    Feature: Support CellDict in graph mode.
    Description: Return the keys of CellDict.
    Expectation: Return expectation expected result.
    """

    class CellDictNet(nn.Cell):
        def __init__(self):
            super(CellDictNet, self).__init__()
            self.cell_dict = nn.CellDict([['conv', nn.Conv2d(6, 16, 5, pad_mode='valid')],
                                          ['relu', nn.ReLU()],
                                          ['max_pool2d', nn.MaxPool2d(kernel_size=4, stride=4)]]
                                         )

        @jit(mode="PIJit")
        def construct(self):
            return self.cell_dict.keys()

    context.set_context(mode=context.PYNATIVE_MODE)
    net = CellDictNet()
    expect_keys = ['conv', 'relu', 'max_pool2d']
    cell_dict_keys = net()
    for key, expect_key in zip(cell_dict_keys, expect_keys):
        assert key == expect_key


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cell_dict_get_values():
    """
    Feature: Support CellDict in graph mode.
    Description: Return the values of CellDict.
    Expectation: Return expectation expected result.
    """

    class CellDictNet(nn.Cell):
        def __init__(self):
            super(CellDictNet, self).__init__()
            self.cell_dict = nn.CellDict([['conv', nn.Conv2d(6, 16, 5, pad_mode='valid')],
                                          ['relu', nn.ReLU()],
                                          ['max_pool2d', nn.MaxPool2d(kernel_size=4, stride=4)]]
                                         )

        @jit(mode="PIJit")
        def construct(self, x):
            outputs = ()
            for cell in self.cell_dict.values():
                res = cell(x)
                outputs = outputs + (res,)
            return outputs

    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor(np.ones([1, 6, 16, 5]), ms.float32)
    cell_dict = {'conv': nn.Conv2d(6, 16, 5, pad_mode='valid'), 'relu': nn.ReLU(),
                 'max_pool2d': nn.MaxPool2d(kernel_size=4, stride=4)}
    expect_res = ()
    for cell in cell_dict.values():
        expect_res = expect_res + (cell(x),)
    net = CellDictNet()
    outputs = net(x)
    for expect, output in zip(expect_res, outputs):
        assert np.allclose(expect.shape, output.shape)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cell_dict_get_items():
    """
    Feature: Support CellDict in graph mode.
    Description: Return the items of CellDict.
    Expectation: Return expectation expected result.
    """

    class CellDictNet(nn.Cell):
        def __init__(self):
            super(CellDictNet, self).__init__()
            self.cell_dict = nn.CellDict([['conv', nn.Conv2d(6, 16, 5, pad_mode='valid')],
                                          ['relu', nn.ReLU()],
                                          ['max_pool2d', nn.MaxPool2d(kernel_size=4, stride=4)]]
                                         )

        @jit(mode="PIJit")
        def construct(self, x):
            key_outputs = ()
            res_outputs = ()
            for key, cell in self.cell_dict.items():
                key_outputs = key_outputs + (key,)
                res = cell(x)
                res_outputs = res_outputs + (res,)
            return key_outputs, res_outputs

    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor(np.ones([1, 6, 16, 5]), ms.float32)
    cell_dict = {'conv': nn.Conv2d(6, 16, 5, pad_mode='valid'), 'relu': nn.ReLU(),
                 'max_pool2d': nn.MaxPool2d(kernel_size=4, stride=4)}
    expect_res = ()
    for cell in cell_dict.values():
        expect_res = expect_res + (cell(x),)
    net = CellDictNet()
    outputs = net(x)
    for expect_key, output_key in zip(cell_dict.keys(), outputs[0]):
        assert expect_key == output_key
    for expect, output in zip(expect_res, outputs[1]):
        assert np.allclose(expect.shape, output.shape)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cell_dict_duplicated_parameter():
    """
    Feature: Support CellDict in graph mode.
    Description: Return the op result of CellDicts with duplicated parameter.
    Expectation: Return expectation expected result.
    """

    class CellDictNet(nn.Cell):
        def __init__(self):
            super(CellDictNet, self).__init__()
            self.cell_dict1 = nn.CellDict([['conv', nn.Conv2d(6, 16, 5, pad_mode='valid')],
                                           ['dense', nn.Dense(3, 4)]]
                                          )
            self.cell_dict2 = nn.CellDict([['conv', nn.Conv2d(6, 16, 5, pad_mode='valid')],
                                           ['dense', nn.Dense(3, 4)]]
                                          )

        @jit(mode="PIJit")
        def construct(self, key1, x1, key2, x2):
            a = self.cell_dict1[key1](x1)
            b = self.cell_dict2[key2](x2)
            return a + b

    context.set_context(mode=context.PYNATIVE_MODE)
    net = CellDictNet()
    x1 = Tensor(np.array([[180, 234, 154], [244, 48, 247]]), ms.float32)
    x2 = Tensor(np.array([[110, 134, 150], [224, 148, 347]]), ms.float32)
    output = net("dense", x1, "dense", x2)
    expect_output_shape = (2, 4)
    assert np.allclose(output.shape, expect_output_shape)
