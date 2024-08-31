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
import os
import sys
import tempfile
import shutil
import numpy as np
import time
import mindspore.context as context
import csv

import mindspore
import mindspore.nn as nn
from mindspore import Tensor, _data_dump, Callback, dataset, Model
from mindspore.ops import operations as P
from mindspore.nn import Cell
from mindspore.nn import Dense
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.nn import Momentum
from mindspore.nn import TrainOneStepCell
from mindspore.nn import WithLossCell
from dump_test_utils import generate_dump_json, generate_dump_json_with_overflow, generate_statistic_dump_json, \
    check_ge_dump_structure, check_ge_dump_structure_acl
from tests.mark_utils import arg_mark
from tests.security_utils import security_off_wrap


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.add = P.Add()

    def construct(self, x_, y_):
        return self.add(x_, y_)


class NetMul(nn.Cell):
    def __init__(self):
        super(NetMul, self).__init__()
        self.mul = P.Mul()

    def construct(self, x_, y_):
        return self.mul(x_, y_)


class NetMulAdd(nn.Cell):
    def __init__(self):
        super(NetMulAdd, self).__init__()
        self.add = P.Add()
        self.mul = P.Mul()

    def construct(self, x_, y_):
        x_ = self.mul(x_, 2)
        y_ = self.mul(y_, 2)
        x_ = self.add(x_, y_)
        y_ = self.add(x_, y_)
        return self.add(x_, y_)


x = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
y = np.array([[7, 8, 9], [10, 11, 12]]).astype(np.float32)


def run_ge_dump(test_name):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_level="O2")
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'ge_dump')
        dump_config_path = os.path.join(tmp_dir, 'ge_dump.json')
        generate_dump_json(dump_path, dump_config_path, test_name)
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        os.environ['ENABLE_MS_GE_DUMP'] = "1"
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)
        add = Net()
        output = add(Tensor(x), Tensor(y))
        check_ge_dump_structure(dump_path, 1, 1)
        if test_name == "test_ge_dump_npy":
            find_x_cmd = 'find {0} -name "Data.x*.output.*.npy"'.format(dump_path)
            x_file_path = os.popen(find_x_cmd).read()
            x_file_path = x_file_path.replace('\n', '')
            find_y_cmd = 'find {0} -name "Data.y*.output.*.npy"'.format(dump_path)
            y_file_path = os.popen(find_y_cmd).read()
            y_file_path = y_file_path.replace('\n', '')
            find_add_cmd = 'find {0} -name "Add.*.output.*.npy"'.format(dump_path)
            add_file_path = os.popen(find_add_cmd).read()
            add_file_path = add_file_path.replace('\n', '')
            x_output = np.load(x_file_path)
            y_output = np.load(y_file_path)
            add_output = np.load(add_file_path)
            assert (x_output == x).all()
            assert (y_output == y).all()
            assert (add_output == output.asnumpy()).all()
        del os.environ['MINDSPORE_DUMP_CONFIG']
        del os.environ['ENABLE_MS_GE_DUMP']


def run_ge_dump_complex(test_name, dtype, enable_ge_dump=True):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_level="O2")
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'ge_dump')
        dump_config_path = os.path.join(tmp_dir, 'ge_dump.json')
        generate_dump_json(dump_path, dump_config_path, test_name)
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        if enable_ge_dump:
            os.environ['ENABLE_MS_GE_DUMP'] = "1"
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)
        add = Net()
        x_complex = np.array([1 + 2j])
        y_complex = np.array([2 + 3j])
        if dtype == "complex64":
            output = add(Tensor(x_complex, mindspore.complex64), Tensor(y_complex, mindspore.complex64))
        else:
            output = add(Tensor(x_complex, mindspore.complex128), Tensor(y_complex, mindspore.complex128))
        find_x_cmd = 'find {0} -name "Data.x*.output.*.npy"'.format(dump_path)
        x_file_path = os.popen(find_x_cmd).read()
        x_file_path = x_file_path.replace('\n', '')
        find_y_cmd = 'find {0} -name "Data.y*.output.*.npy"'.format(dump_path)
        y_file_path = os.popen(find_y_cmd).read()
        y_file_path = y_file_path.replace('\n', '')
        find_add_cmd = 'find {0} -name "Add.*.output.*.npy"'.format(dump_path)
        add_file_path = os.popen(find_add_cmd).read()
        add_file_path = add_file_path.replace('\n', '')
        x_output = np.load(x_file_path)
        y_output = np.load(y_file_path)
        add_output = np.load(add_file_path)
        assert (x_output == x_complex).all()
        assert (y_output == y_complex).all()
        assert (add_output == output.asnumpy()).all()
        del os.environ['MINDSPORE_DUMP_CONFIG']
        if enable_ge_dump:
            del os.environ['ENABLE_MS_GE_DUMP']


def run_ge_dump_acl(test_name):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_level="O2")
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'acl_dump')
        dump_config_path = os.path.join(tmp_dir, 'acl_dump.json')
        generate_dump_json(dump_path, dump_config_path, test_name)
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)
        os.mkdir(dump_path)
        add = Net()
        add(Tensor(x), Tensor(y))
        check_ge_dump_structure_acl(dump_path, 0, 1)
        del os.environ['MINDSPORE_DUMP_CONFIG']


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@security_off_wrap
def test_ge_dump():
    """
    Feature: async dump on Ascend on GE backend.
    Description: test async dump with default file_format value ("bin")
    Expectation: dump data are generated as protobuf file format (suffix with timestamp)
    """
    run_ge_dump("test_ge_dump")


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@security_off_wrap
def test_ge_dump_acl():
    """
    Feature: async dump on Ascend on GE backend.
    Description: test async dump with default file_format value ("bin")
    Expectation: dump data are generated as protobuf file format (suffix with timestamp)
    """
    run_ge_dump_acl("test_acl_dump")


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@security_off_wrap
def test_ge_dump_acl_assign_ops_by_regex():
    """
    Feature: async dump on Ascend on GE backend.
    Description: test async dump with default file_format value ("bin")
    Expectation: dump data are generated as protobuf file format (suffix with timestamp)
    """
    run_ge_dump_acl("test_acl_dump_assign_ops_by_regex")


class ReluReduceMeanDenseRelu(Cell):
    def __init__(self, kernel, bias, in_channel, num_class):
        super().__init__()
        self.relu = P.ReLU()
        self.mean = P.ReduceMean(keep_dims=False)
        self.dense = Dense(in_channel, num_class, kernel, bias)

    def construct(self, x_):
        x_ = self.relu(x_)
        x_ = self.mean(x_, (2, 3))
        x_ = self.dense(x_)
        x_ = self.relu(x_)
        return x_


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@security_off_wrap
def test_ge_dump_net_multi_layer_mode1():
    """
    Feature: async dump on Ascend on GE backend.
    Description: test async dump on GE backend.
    Expectation: dump data are generated as GE dump structure.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'ge_dump_net_multi_layer_mode1')
        json_file_path = os.path.join(tmp_dir, "test_ge_dump_net_multi_layer_mode1.json")
        generate_dump_json(dump_path, json_file_path, 'test_ge_dump_net_multi_layer_mode1', 'test')
        os.environ['MINDSPORE_DUMP_CONFIG'] = json_file_path
        os.environ['ENABLE_MS_GE_DUMP'] = "1"
        weight = Tensor(np.ones((1000, 2048)).astype(np.float32))
        bias = Tensor(np.ones((1000,)).astype(np.float32))
        net = ReluReduceMeanDenseRelu(weight, bias, 2048, 1000)
        criterion = SoftmaxCrossEntropyWithLogits(sparse=False)
        optimizer = Momentum(learning_rate=0.1, momentum=0.1,
                             params=filter(lambda x: x.requires_grad, net.get_parameters()))
        net_with_criterion = WithLossCell(net, criterion)
        train_network = TrainOneStepCell(net_with_criterion, optimizer)
        train_network.set_train()
        inputs = Tensor(np.random.randn(32, 2048, 7, 7).astype(np.float32))
        label = Tensor(np.zeros(shape=(32, 1000)).astype(np.float32))
        _ = train_network(inputs, label)
        check_ge_dump_structure(dump_path, 1, 1)
        del os.environ['MINDSPORE_DUMP_CONFIG']
        del os.environ['ENABLE_MS_GE_DUMP']


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@security_off_wrap
def test_ge_dump_with_diagnostic_path():
    """
    Feature: Dump on GE backend when the MS_DIANOSTIC_DATA_PATH is set.
    Description: Test Ascend dump on GE when path is not set (set to empty) in dump json file and
     MS_DIAGNOSTIC_DATA_PATH is set.
    Expectation: Data is expected to be dumped into MS_DIAGNOSTIC_DATA_PATH/debug_dump.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_config_path = os.path.join(tmp_dir, 'ge_dump.json')
        generate_dump_json('', dump_config_path, 'test_ge_dump')
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        os.environ['ENABLE_MS_GE_DUMP'] = "1"
        diagnose_path = os.path.join(tmp_dir, 'ge_dump')
        os.environ['MS_DIAGNOSTIC_DATA_PATH'] = diagnose_path
        add = Net()
        add(Tensor(x), Tensor(y))
        dump_path = os.path.join(diagnose_path, 'debug_dump')
        check_ge_dump_structure(dump_path, 1, 1)
        del os.environ['MINDSPORE_DUMP_CONFIG']
        del os.environ['MS_DIAGNOSTIC_DATA_PATH']
        del os.environ['ENABLE_MS_GE_DUMP']


def run_overflow_dump():
    """Run async dump and generate overflow"""
    if sys.platform != 'linux':
        return
    context.set_context(jit_level="O2")
    overflow_x = np.array([60000, 60000]).astype(np.float16)
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'overflow_dump')
        dump_config_path = os.path.join(tmp_dir, 'overflow_dump.json')
        generate_dump_json_with_overflow(dump_path, dump_config_path, 'test_ge_dump', 3)
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        os.environ['ENABLE_MS_GE_DUMP'] = "1"
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)
        add = Net()
        output = add(Tensor(overflow_x), Tensor(overflow_x))
        output_np = output.asnumpy()
        print("output_np: ", output_np)
        check_ge_dump_structure(dump_path, 1, 1, True)
        del os.environ['MINDSPORE_DUMP_CONFIG']
        del os.environ['ENABLE_MS_GE_DUMP']


def run_set_overflow_dump():
    """Run async dump and generate overflow"""
    if sys.platform != 'linux':
        return
    context.set_context(jit_level="O2")
    data = np.array([60000, 60000]).astype(np.float16)
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'overflow_dump')
        dump_config_path = os.path.join(tmp_dir, 'overflow_dump.json')
        set_overflow_num = 1
        generate_dump_json_with_overflow(dump_path, dump_config_path, 'test_ge_dump_npy', 3, set_overflow_num)
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        os.environ['ENABLE_MS_GE_DUMP'] = "1"
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)
        net = NetMulAdd()
        net(Tensor(data), Tensor(data))

        check_ge_dump_structure(dump_path, 1, 1, True, check_overflow_num=True, overflow_number=set_overflow_num)
        del os.environ['MINDSPORE_DUMP_CONFIG']
        del os.environ['ENABLE_MS_GE_DUMP']


def run_saved_data_dump_test_bf16(scenario, saved_data):
    """Run dump on GE backend, testing statistic dump"""
    if sys.platform != 'linux':
        return
    with tempfile.TemporaryDirectory() as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'test_saved_data')
        dump_config_path = os.path.join(tmp_dir, 'test_saved_data.json')
        generate_statistic_dump_json(dump_path, dump_config_path, scenario, saved_data)
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        os.environ['ENABLE_MS_GE_DUMP'] = "1"
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)

        x_np = np.array([1.1, 2.3, 3.7])
        y_np = np.array([4.1, 5.3, 6.7])
        x_tensor = Tensor(x_np, mindspore.bfloat16)
        y_tensor = Tensor(y_np, mindspore.bfloat16)

        mul = NetMul()
        mul(x_tensor, y_tensor)

        for _ in range(3):
            if not os.path.exists(dump_path):
                time.sleep(2)
        find_x_cmd = 'find {0} -name "Data.x_*.output.*.npy"'.format(dump_path)
        x_file_path = os.popen(find_x_cmd).read()
        x_file_path = x_file_path.replace('\n', '')
        statistic_cmd = 'find {0} -name "statistic.csv"'.format(dump_path)
        statistic_file_path = os.popen(statistic_cmd).read()
        statistic_files_path = statistic_file_path.split('\n')

        x_output = np.load(x_file_path)
        x_float32 = Tensor(x_tensor, mindspore.float32).asnumpy()
        assert (x_float32 == x_output).all()
        with open(statistic_files_path[0], 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            _ = next(csv_reader)
            for row in csv_reader:
                if row[0] == "Data" and row[1] == "x_":
                    assert row[8] == "bfloat16"
                    break
        del os.environ['MINDSPORE_DUMP_CONFIG']
        del os.environ['ENABLE_MS_GE_DUMP']


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_ge_statistic_dump_bfloat16():
    """
    Feature: Ascend Statistics Dump on GE backend
    Description: Test Ascend statistics dump
    Expectation: Statistics are stored in statistic.csv files
    """

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_level="O2")
    run_saved_data_dump_test_bf16('test_ge_dump', 'full')


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@security_off_wrap
def test_ge_overflow_dump():
    """
    Feature: Overflow Dump on GE backend
    Description: Test overflow dump
    Expectation: Overflow is occurred, and overflow dump file is in correct format
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
    context.set_context(jit_level="O2")
    run_overflow_dump()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@security_off_wrap
def test_ge_set_overflow_dump():
    """
    Feature: The number of overflow dump can be configured on GE backend
    Description: Test set overflow dump number
    Expectation: Overflow is occurred, and the number of overflow dump file is in correct
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
    context.set_context(jit_level="O2")
    run_set_overflow_dump()


def run_train():
    context.set_context(mode=context.GRAPH_MODE)
    context.set_context(jit_level="O2")
    add = Net()
    add(Tensor(x), Tensor(y))


def run_saved_data_dump_test(scenario, saved_data, enable_ge_dump=True):
    """Run dump on GE backend, testing statistic dump"""
    if sys.platform != 'linux':
        return
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'test_saved_data')
        dump_config_path = os.path.join(tmp_dir, 'test_saved_data.json')
        generate_statistic_dump_json(dump_path, dump_config_path, scenario, saved_data)
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        if enable_ge_dump:
            os.environ['ENABLE_MS_GE_DUMP'] = "1"
        exec_network_cmd = 'cd {0}; python -c "from test_ge_dump import run_train; run_train()"'.format(os.getcwd())
        _ = os.system(exec_network_cmd)
        if enable_ge_dump:
            check_ge_dump_structure(dump_path, 1, 1, saved_data=saved_data)
        else:
            check_ge_dump_structure_acl(dump_path, 0, 1, False, saved_data)
        del os.environ['MINDSPORE_DUMP_CONFIG']
        if enable_ge_dump:
            del os.environ['ENABLE_MS_GE_DUMP']


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@security_off_wrap
def test_ge_statistic_dump():
    """
    Feature: Ascend Statistics Dump on GE backend
    Description: Test Ascend statistics dump
    Expectation: Statistics are stored in statistic.csv files
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_saved_data_dump_test('test_ge_dump', 'statistic')


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@security_off_wrap
def test_ge_tensor_dump():
    """
    Feature: Ascend Tensor Dump on GE backend
    Description: Test Ascend tensor dump
    Expectation: Tensors are stored in npy files
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_saved_data_dump_test('test_ge_dump', 'tensor')


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@security_off_wrap
def test_ge_full_dump():
    """
    Feature: Ascend Full Dump on GE backend
    Description: Test Ascend full dump
    Expectation: Tensors are stored in npy files and their statistics stored in statistic.csv
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_saved_data_dump_test('test_ge_dump', 'full')


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@security_off_wrap
def test_ge_dump_npy():
    """
    Feature: async dump on Ascend on GE backend.
    Description: test async dump with file_format set to npy
    Expectation: dump data are generated as npy files, and the value is correct
    """
    run_ge_dump("test_ge_dump_npy")


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@security_off_wrap
def test_ge_dump_complex64():
    """
    Feature: async dump on Ascend on GE backend.
    Description: test async dump with file_format set to npy in complex64 dtype
    Expectation: dump data are generated as npy files, and the value is correct
    """
    run_ge_dump_complex("test_ge_dump_npy", "complex64")


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@security_off_wrap
def test_ge_dump_complex128():
    """
    Feature: async dump on Ascend on GE backend.
    Description: test async dump with file_format set to npy in complex128 dtype
    Expectation: dump data are generated as npy files, and the value is correct
    """
    run_ge_dump_complex("test_ge_dump_npy", "complex128")


class DynamicNet(Cell):
    def __init__(self):
        """init"""
        super(DynamicNet, self).__init__()
        self.relu = nn.ReLU()
        self.cast = P.Cast()
        self.expanddim = P.ExpandDims()


    def construct(self, x_):
        """construct."""
        y_ = self.expanddim(x_, 1)
        out = self.relu(y_)
        return out


def train_dynamic_net():
    mindspore.set_context(mode=mindspore.GRAPH_MODE)
    net = DynamicNet()
    input_dyn = Tensor(shape=[3, None], dtype=mindspore.float32)
    net.set_inputs(input_dyn)
    input1 = Tensor(np.random.random([3, 10]), dtype=mindspore.float32)
    _ = net(input1)


def run_ge_dump_acl_dynamic_shape(test_name):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'acl_dump_dynamic_shape')
        dump_config_path = os.path.join(tmp_dir, 'acl_dump.json')
        generate_dump_json(dump_path, dump_config_path, test_name)
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)
        os.mkdir(dump_path)
        train_dynamic_net()
        check_ge_dump_structure_acl(dump_path, 0, 1, saved_data="tensor", is_kbk=True)
        del os.environ['MINDSPORE_DUMP_CONFIG']


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@security_off_wrap
def test_ge_dump_acl_dynamic_shape():
    """
    Feature: async dump on Ascend on GE backend.
    Description: test async dump with default file_format value ("bin")
    Expectation: dump data are generated as protobuf file format (suffix with timestamp)
    """
    run_ge_dump_acl_dynamic_shape("test_acl_dump_dynamic_shape")


def run_overflow_acl_dump():
    """Run async dump and generate overflow"""
    if sys.platform != 'linux':
        return
    context.set_context(mode=mindspore.GRAPH_MODE, jit_level="O2")
    overflow_x = np.array([60000, 60000]).astype(np.float16)
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'overflow_dump')
        dump_config_path = os.path.join(tmp_dir, 'overflow_dump.json')
        generate_dump_json_with_overflow(dump_path, dump_config_path, 'test_overflow_acl_dump', 3)
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)
        add = Net()
        output = add(Tensor(overflow_x), Tensor(overflow_x))
        output_np = output.asnumpy()
        print("output_np: ", output_np)
        check_ge_dump_structure_acl(dump_path, 0, 1, True)
        del os.environ['MINDSPORE_DUMP_CONFIG']


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@security_off_wrap
def test_overflow_acl_dump():
    """
    Feature: overflow dump using acl dump.
    Description: test acl dump with  overflow dump enabled.
    Expectation:overflow  dump data are generated.
    """
    run_overflow_acl_dump()


def run_acl_set_dump():
    """Run acl dump and generate data with dump flag"""
    if sys.platform != 'linux':
        return
    context.set_context(mode=mindspore.GRAPH_MODE, jit_level="O2")
    data = np.array([60, 60]).astype(np.float16)
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'acl_set_dump')
        dump_config_path = os.path.join(tmp_dir, 'acl_set_dump.json')
        generate_dump_json(dump_path, dump_config_path, 'test_acl_set_dump')
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)
        net = NetMulAdd()
        mindspore.set_dump(net.add)
        net(Tensor(data), Tensor(data))
        check_ge_dump_structure_acl(dump_path, 0, 1)
        del os.environ['MINDSPORE_DUMP_CONFIG']


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@security_off_wrap
def test_acl_set_dump():
    """
    Feature: O2 process supports set dump.
    Description: test acl dump with set dump enabled.
    Expectation: Using mindpore.set_dump to enable dump functionality on objects will generate data.
    """
    run_acl_set_dump()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@security_off_wrap
def test_acl_statistic_dump():
    """
    Feature: Ascend Statistics Dump on GE backend
    Description: Test Ascend statistics dump
    Expectation: Statistics are stored in statistic.csv files
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_saved_data_dump_test('test_acl_dump', 'statistic', False)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@security_off_wrap
def test_acl_tensor_dump():
    """
    Feature: Ascend Tensor Dump on GE backend
    Description: Test Ascend tensor dump
    Expectation: Tensors are stored in npy files
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_saved_data_dump_test('test_acl_dump', 'tensor', False)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@security_off_wrap
def test_acl_full_dump():
    """
    Feature: Ascend Full Dump on GE backend
    Description: Test Ascend full dump
    Expectation: Tensors are stored in npy files and their statistics stored in statistic.csv
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    run_saved_data_dump_test('test_acl_dump', 'full', False)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@security_off_wrap
def test_acl_dump_complex64():
    """
    Feature:acl dump on Ascend.
    Description: test acl dump with file_format set to npy in complex64 dtype
    Expectation: dump data are generated as npy files, and the value is correct
    """
    run_ge_dump_complex("test_acl_dump_complex", "complex64", False)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@security_off_wrap
def test_acl_dump_complex128():
    """
    Feature: acl dump on Ascend.
    Description: test acl dump with file_format set to npy in complex128 dtype
    Expectation: dump data are generated as npy files, and the value is correct
    """
    run_ge_dump_complex("test_acl_dump_complex", "complex128", False)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@security_off_wrap
def test_acl_dump_with_diagnostic_path():
    """
    Feature: acl dump  when the MS_DIANOSTIC_DATA_PATH is set.
    Description: Test acl dump when path is not set (set to empty) in dump json file and
     MS_DIAGNOSTIC_DATA_PATH is set.
    Expectation: Data is expected to be dumped into MS_DIAGNOSTIC_DATA_PATH/debug_dump.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_level="O2")
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_config_path = os.path.join(tmp_dir, 'acl_dump.json')
        generate_dump_json('', dump_config_path, 'test_acl_dump')
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        diagnose_path = os.path.join(tmp_dir, 'acl_dump')
        os.environ['MS_DIAGNOSTIC_DATA_PATH'] = diagnose_path
        add = Net()
        add(Tensor(x), Tensor(y))
        dump_path = os.path.join(diagnose_path, 'debug_dump')
        check_ge_dump_structure_acl(dump_path, 0, 1)
        del os.environ['MINDSPORE_DUMP_CONFIG']
        del os.environ['MS_DIAGNOSTIC_DATA_PATH']

class StopAtStep(Callback):
    """
    Start profiling base on step.

    Args:
        start_step (int): The start step number.
        stop_step (int): The stop step number.
    """
    def __init__(self, start_step, stop_step):
        super(StopAtStep, self).__init__()
        self.start_step = start_step
        self.stop_step = stop_step
        # pylint: disable=W0212
        _data_dump._dump_set_dynamic()

    def on_train_step_begin(self, run_context):
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        if step_num == self.start_step:
            # pylint: disable=W0212
            _data_dump._dump_start()

    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        step_num = cb_params.cur_step_num
        if step_num == self.stop_step:
            # pylint: disable=W0212
            _data_dump._dump_stop()

class Net1(nn.Cell):
    """The test net"""
    def __init__(self):
        super(Net1, self).__init__()
        self.fc = nn.Dense(2, 2)

    def construct(self, x_):
        return self.fc(x_)

def generator():
    for _ in range(3):
        yield (np.ones([2, 2]).astype(np.float32), np.ones([2]).astype(np.int32))

def run_dynamic_acl_dump_flag(test_name):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_level="O2")
    if sys.platform != 'linux':
        return
    with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'acl_dump')
        dump_config_path = os.path.join(tmp_dir, 'acl_dump.json')
        generate_dump_json(dump_path, dump_config_path, test_name)
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path
        dynamic_data_dump = StopAtStep(2, 3)
        if os.path.isdir(dump_path):
            shutil.rmtree(dump_path)
        network = Net1()
        optimizer = nn.Momentum(network.trainable_params(), 1, 0.9)
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        data = dataset.GeneratorDataset(generator, ["data", "label"])
        model = Model(network, loss, optimizer)
        model.train(3, data, callbacks=[dynamic_data_dump], dataset_sink_mode=False)
        dump_data_path = os.path.join(dump_path, '2')
        assert os.path.exists(dump_data_path)
        del os.environ['MINDSPORE_DUMP_CONFIG']

@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@security_off_wrap
def test_acl_dynamic_dump():
    """
    Feature: acl dump  when the MS_DIANOSTIC_DATA_PATH is set.
    Description: Test acl dump when path is not set (set to empty) in dump json file and
     MS_DIAGNOSTIC_DATA_PATH is set.
    Expectation: Data is expected to be dumped into MS_DIAGNOSTIC_DATA_PATH/debug_dump.
    """
    run_dynamic_acl_dump_flag('test_acl_dump')
