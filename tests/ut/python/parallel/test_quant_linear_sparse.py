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

import numpy as np

import mindspore as ms
from mindspore import context, Tensor, Parameter
from mindspore.nn import Cell
from mindspore.ops.operations._infer_ops import QuantLinearSparse
from parallel.utils.utils import ParallelValidator, compile_net


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class QuantLinearSparseNet(Cell):
    def __init__(self, strategy):
        super().__init__()
        self.quant_linear_sparse = QuantLinearSparse().shard(strategy)

    def construct(self, x, weight, deqScale, compressIdx, bias):
        return self.quant_linear_sparse(x, weight, deqScale, compressIdx, bias)


def test_quant_linear_sparse():
    """
    Feature: test quant linear sparse semi
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)

    strategy = ((1, 2), (1,), (2,), (1,), (2,))

    net = QuantLinearSparseNet(strategy)

    m = 4096
    k = 4096
    n = 4096

    x = Parameter(Tensor(np.ones([m, k]), dtype=ms.int8), "x")
    weight_np = np.ones([n])
    weight = Parameter(Tensor(weight_np, dtype=ms.int8), "weight")
    deqScale = Parameter(Tensor(np.ones([n]), dtype=ms.int64), "deqScale")
    compressIdx_np = np.ones([n])
    compressIdx = Parameter(Tensor(compressIdx_np, dtype=ms.int8), "compressIdx")
    bias = Parameter(Tensor(np.ones([n]), dtype=ms.int32), "bias")

    net.set_inputs(x, weight, deqScale, compressIdx, bias)

    phase = compile_net(net, x, weight, deqScale, compressIdx, bias)
    validator = ParallelValidator(net, phase)

    assert validator.check_parameter_shape("x", [m, k//2])
    assert validator.check_parameter_shape("deqScale", [n//2])
    assert validator.check_parameter_shape("bias", [n//2])
