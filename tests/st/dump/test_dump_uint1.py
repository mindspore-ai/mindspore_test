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
import csv
import tempfile
import time
import numpy as np
import mindspore.nn as nn
from mindspore.common.initializer import Normal
import mindspore.context as context
from mindspore.train import Model
from mindspore.dataset import GeneratorDataset
from dump_test_utils import generate_statistic_dump_json
from tests.mark_utils import arg_mark
from tests.security_utils import security_off_wrap


class LeNet5(nn.Cell):
    """
    Lenet network

    Args:
        num_class (int): Number of classes. Default: 10.
        num_channel (int): Number of channels. Default: 1.

    Returns:
        Tensor, output tensor
    Examples:
        >>> LeNet(num_class=10)

    """
    def __init__(self, num_class=10, num_channel=1, include_top=True):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.include_top = include_top
        if self.include_top:
            self.flatten = nn.Flatten()
            self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
            self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
            self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))


    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        if not self.include_top:
            return x
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CustomDataset:
    """
    Generate custom dataset.
    """
    def __init__(self):
        self._data = np.ones([2, 2, 1, 32, 32])
        self._label = np.zeros([2, 2, 10])
    def __getitem__(self, index):
        return self._data[index], self._label[index]
    def __len__(self):
        return len(self._data)


def run_dump_uint1():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", jit_config={"jit_level": "O2"})
    with tempfile.TemporaryDirectory() as tmp_dir:
        dump_path = os.path.join(tmp_dir, 'test_dump_uint1')
        dump_config_path = os.path.join(tmp_dir, 'test_dump_uint1.json')
        generate_statistic_dump_json(dump_path, dump_config_path, 'test_async_dump', 'full')
        os.environ['MINDSPORE_DUMP_CONFIG'] = dump_config_path

        loader = CustomDataset()
        dataset = GeneratorDataset(source=loader, column_names=["data", "label"])
        net = LeNet5()
        loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
        net_opt = nn.Momentum(net.trainable_params(), 0.002, 0.9)
        model = Model(net, loss_fn, net_opt, amp_level="O2")
        model.train(1, dataset, dataset_sink_mode=True, sink_size=1)

        for _ in range(3):
            if not os.path.exists(dump_path):
                time.sleep(2)
        find_uint1_output_cmd = 'find {0} -name "ReluV2*.*op1.*.output.1.*.npy"'.format(dump_path)
        uint1_output_file_path = os.popen(find_uint1_output_cmd).read()
        uint1_output_file_path = uint1_output_file_path.replace('\n', '')
        dump_uint1 = np.load(uint1_output_file_path)
        assert uint1_output_file_path.endswith(".uint1.npy")
        assert dump_uint1.dtype == np.uint8

        find_statistic_cmd = 'find {0} -name "statistic.csv"'.format(dump_path)
        statistic_file = os.popen(find_statistic_cmd).read()
        statistic_file = statistic_file.replace('\n', '')
        with open(statistic_file) as f:
            reader = csv.DictReader(f)
            stats = list(reader)

            def get_uint1_data(statistic):
                return statistic['Data Type'] == 'uint1'

            uint1_statistics = list(filter(get_uint1_data, stats))
            uint1_num = len(uint1_statistics)
            assert uint1_num == 4
            for statistic_item in uint1_statistics:
                assert statistic_item['Max Value'] == '1' or '0'
                assert statistic_item['Min Value'] == '1' or '0'


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@security_off_wrap
def test_acl_dump_uint1():
    """
    Feature: acl dump  for uint1 data type.
    Description: Test acl dump  when the output data is with dtype uint1.
    Expectation: Data is expected to be dumped correctly, and the statistic file is correctly record
     the data type and  the statistic items.
    """
    run_dump_uint1()
