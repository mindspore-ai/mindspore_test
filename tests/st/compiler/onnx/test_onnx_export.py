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
""" Test onnx export"""
import os
import numpy as np
import onnxruntime as ort
import mindspore as ms
from mindspore import nn
from mindspore import Tensor, ops, Parameter
from mindspore.onnx import export
from tests.mark_utils import arg_mark


class NetGeLU(nn.Cell):
    def __init__(self, mul_size):
        super().__init__()
        self.op = ops.operations.GeLU()
        mul_np = np.full(mul_size, 2.0, dtype=np.float32)
        self.mul_weight = Parameter(Tensor(mul_np), name="mul_weight")

    def construct(self, inputs):
        out = ops.operations.Mul()(inputs, self.mul_weight)
        out = self.op(out)
        return out


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_ms_onnx_export_01():
    """
    Feature: Onnx
    Description: Test onnx export
    Expectation: No exception
    """
    ms.set_context(mode=ms.GRAPH_MODE)
    _cur_dir = os.path.dirname(os.path.realpath(__file__))
    file_name = os.path.join(_cur_dir, "test_ms_onnx_export_01")
    onnx_file_name = file_name + '.onnx'

    mul_size = (3,)
    input_data = [1.0, 2.0, 3.0]
    ms_input_data = Tensor(input_data, dtype=ms.float32)
    np_input_data = np.array(input_data, dtype=np.float32)

    try:
        net = NetGeLU(mul_size=mul_size)
        ms_output = net(ms_input_data)
        ms_output_numpy = ms_output.numpy()
        export(net, ms_input_data, file_name=onnx_file_name)
        assert os.path.exists(onnx_file_name)

        session = ort.InferenceSession(onnx_file_name)
        onnx_output = session.run(None, {'inputs': np_input_data})
        assert np.allclose(ms_output_numpy, onnx_output[0], 0.001, 0.001)
    finally:
        if os.path.exists(onnx_file_name):
            os.remove(onnx_file_name)
