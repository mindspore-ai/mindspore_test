# Copyright 2019 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class NetSoftmaxWithCrossEntropy(nn.Cell):
    def __init__(self):
        super(NetSoftmaxWithCrossEntropy, self).__init__()
        logits = Tensor(np.array([[1, 1, 10],
                                  [1, 10, 1],
                                  [10, 1, 1]]).astype(np.float32))
        self.logits = Parameter(initializer(logits, logits.shape), name='logits')
        labels = Tensor(np.array([2, 1, 0]).astype(np.int32))
        self.labels = Parameter(initializer(labels, labels.shape), name='labels')
        self.SoftmaxWithCrossEntropy = P.SparseSoftmaxCrossEntropyWithLogits(True)

    def construct(self):
        return self.SoftmaxWithCrossEntropy(self.logits, self.labels)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='essential')
def test_net():
    SoftmaxWithCrossEntropy = NetSoftmaxWithCrossEntropy()
    output = SoftmaxWithCrossEntropy()
    expect = np.array([[4.1126452e-05, 4.1126452e-05, -8.2234539e-05],
                       [4.1126452e-05, -8.2234539e-05, 4.1126452e-05],
                       [-8.2234539e-05, 4.1126452e-05, 4.1126452e-05]]).astype(np.float32)
    print(output)
    error = np.ones(shape=[3, 3]) * 1.0e-6
    diff = output.asnumpy() - expect
    print(diff)
    assert np.all(diff < error)
    assert np.all(-diff < error)
