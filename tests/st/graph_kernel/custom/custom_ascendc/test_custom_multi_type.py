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

import pytest
import numpy as np
import mindspore as ms
from mindspore.nn import Cell
import mindspore.ops as ops
from mindspore import context, Tensor
from mindspore.ops import DataType, CustomRegOp
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x, dim=None, keepdim=False):
    return np.argmin(x, axis=dim)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE])
def test_custom_argmin(context_mode):
    """
    Feature: custom op multi type.
    Description: test function argmin forward.
    Expectation: expect correct result.
    """

    class CustomNet(Cell):
        def __init__(self):
            super(CustomNet, self).__init__()
            aclnn_reg_info = CustomRegOp("aclnnArgMin") \
                .input(0, "x", "required") \
                .input(1, "y", "required") \
                .input(2, "w", "required") \
                .output(0, "z", "required") \
                .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
                .target("Ascend") \
                .get_op_info()

            self.custom_argmin = ops.Custom("./infer_file/custom_callback.cc:aclnnArgMin",
                                            lambda x, y, z: (x[1], x[2], x[3]), ms.dtype.int64,
                                            func_type="aot", bprop=None,
                                            reg_info=aclnn_reg_info)
            self.custom_argmin.add_prim_attr("custom_inputs_type", "tensor,int,bool")

        def construct(self, x, y, z):
            res = self.custom_argmin(x, y, z)
            return res

    context.set_context(mode=context_mode, save_graphs=False, save_graphs_path="./graphs",
                        jit_config={"jit_level": "O0"})
    x = generate_random_input((2, 3, 4, 5), np.float32)
    dim = 0
    keep_dim = False

    net = CustomNet()
    output = net(ms.Tensor(x), dim, keep_dim)
    expect = generate_expect_forward_output(x, dim, keep_dim)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE])
def test_custom_mul_aclnn(context_mode):
    """
    Feature: Custom op testcase
    Description: test case for mul by custom
    Expectation: the result match with numpy result
    """

    class CustomNet(Cell):
        def __init__(self, func, out_shape, bprop):
            super(CustomNet, self).__init__()
            aclnn_ref_info = CustomRegOp("aclnnMul") \
                .input(0, "x", "required") \
                .input(1, "y", "required") \
                .output(0, "z", "required") \
                .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
                .target("Ascend") \
                .get_op_info()

            self.custom_mul = ops.Custom(func, out_shape, lambda x, _: x, func_type="aot", bprop=bprop,
                                         reg_info=aclnn_ref_info)
            self.custom_mul.add_prim_attr("custom_inputs_type", "tensor,tensor")
            self.add = P.Add()
            self.sub = P.Sub()

        def construct(self, x, y, z):
            res = self.add(x, y)
            res = self.custom_mul(res, y)
            res = self.sub(res, z)
            return res

    context.set_context(mode=context_mode, save_graphs=False, save_graphs_path="./graphs",
                        jit_config={"jit_level": "O0"})

    x = np.ones([8, 2048]).astype(np.float16)
    y = np.ones([8, 2048]).astype(np.float16)
    z = np.random.rand(8, 2048).astype(np.float16)
    net = CustomNet("./infer_file/custom_callback.cc:aclnnMul", lambda x, _: x, None)
    expect_out = (x + y) * y - z
    out = net(Tensor(x), Tensor(y), Tensor(z))
    assert np.allclose(out.asnumpy(), expect_out, 0.001, 0.001)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level4', card_mark='onecard', essential_mark='essential')
def test_custom_topk():
    """
  Feature: Custom op testcase
  Description: test case for topk by custom
  Expectation: the result match with numpy result
  """

    class MoeSoftMaxTopkNet(Cell):
        def __init__(self, func, out_shape):
            super(MoeSoftMaxTopkNet, self).__init__()
            reg_info = CustomRegOp("MoeSoftMaxTopk") \
                .input(0, "x", "required") \
                .output(0, "y", "required") \
                .output(1, "indices", "required") \
                .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.I32_Default) \
                .target("Ascend") \
                .get_op_info()

            self.moe_softmax_topk_custom = ops.Custom(func=func, out_shape=out_shape,
                                                      out_dtype=[mstype.float32, mstype.int32], func_type="aot",
                                                      bprop=None, reg_info=reg_info)
            self.moe_softmax_topk_custom.add_prim_attr("custom_inputs_type", "tensor,int")

        def construct(self, x):
            res = self.moe_softmax_topk_custom(x, 2)
            return res

    ms.set_context(mode=ms.GRAPH_MODE, save_graphs=False, save_graphs_path="./graph",
                   pynative_synchronize=False)
    ms.set_context(jit_config={"jit_level": "O1"})

    input_x = ms.Tensor(np.ones([1024, 16]), ms.float32)
    # 通过lambda实现infer shape函数
    net = MoeSoftMaxTopkNet("./infer_file/custom_callback.cc:MoeSoftMaxTopk", lambda x, _: [[x[0], 2], [x[0], 2]])
    output = net(input_x)
    print(output)
