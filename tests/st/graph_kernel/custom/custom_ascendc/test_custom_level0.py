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

import pytest
from tests.mark_utils import arg_mark
import numpy as np
import mindspore as ms
from mindspore import context, Tensor
from mindspore.nn import Cell
import mindspore.ops as ops
from mindspore.ops import DataType, CustomRegOp
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype


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
        self.add = P.Add()
        self.sub = P.Sub()

    def construct(self, x, y, z):
        res = self.add(x, y)
        res = self.custom_mul(res, y)
        res = self.sub(res, z)
        return res


class CustomNetAddPrefix(Cell):
    def __init__(self, func, out_shape, bprop):
        super(CustomNetAddPrefix, self).__init__()
        aclnn_ref_info = CustomRegOp("Mul") \
            .input(0, "x", "required") \
            .input(1, "y", "required") \
            .output(0, "z", "required") \
            .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
            .target("Ascend") \
            .get_op_info()

        self.custom_mul = ops.Custom(func, out_shape, lambda x, _: x, func_type="aot", bprop=bprop,
                                     reg_info=aclnn_ref_info)
        self.add = P.Add()
        self.sub = P.Sub()

    def construct(self, x, y, z):
        res = self.add(x, y)
        res = self.custom_mul(res, y)
        res = self.sub(res, z)
        return res


class CustomNetAclOp(Cell):
    def __init__(self, func, out_shape, bprop):
        super(CustomNetAclOp, self).__init__()
        aclnn_ref_info = CustomRegOp("Mul") \
            .input(0, "x", "required") \
            .input(1, "y", "required") \
            .output(0, "z", "required") \
            .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
            .target("Ascend") \
            .get_op_info()

        self.custom_mul = ops.Custom(func, out_shape, lambda x, _: x, func_type="aot", bprop=bprop,
                                     reg_info=aclnn_ref_info)
        self.custom_mul.add_prim_attr("custom_aclop", True)
        self.add = P.Add()
        self.sub = P.Sub()

    def construct(self, x, y, z):
        res = self.add(x, y)
        res = self.custom_mul(res, y)
        res = self.sub(res, z)
        return res


class BaseNet(Cell):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.add = P.Add()
        self.sub = P.Sub()
        self.mul = P.Mul()

    def construct(self, x, y, z):
        res = self.add(x, y)
        res = self.mul(res, y)
        res = self.sub(res, z)
        return res


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_custom_mul_aclnn(context_mode):
    """
    Feature: Custom op testcase
    Description: test case for mul by custom
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context_mode, save_graphs=False, save_graphs_path="./graphs",
                        jit_config={"jit_level": "O0"})
    x = np.ones([8, 2048]).astype(np.float16)
    y = np.ones([8, 2048]).astype(np.float16)
    z = np.random.rand(8, 2048).astype(np.float16)
    net = CustomNet("aclnnMul", lambda x, _: x, None)
    expect_out = (x + y) * y - z
    out = net(Tensor(x), Tensor(y), Tensor(z))
    assert np.allclose(out.asnumpy(), expect_out, 0.001, 0.001)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_custom_mul_aclnn_dynamic(context_mode):
    """
    Feature: Custom op testcase
    Description: test case for mul by custom in dynamic shape
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context_mode, save_graphs=False, save_graphs_path="./graphs",
                        jit_config={"jit_level": "O0"})
    x = np.ones([8, 2048]).astype(np.float16)
    y = np.ones([8, 2048]).astype(np.float16)
    z = np.random.rand(8, 2048).astype(np.float16)
    dyn_x = Tensor(shape=(8, None), dtype=mstype.float16)
    net = CustomNet("aclnnMul", lambda x, _: x, None)
    expect_out = (x + y) * y - z
    net.set_inputs(dyn_x, Tensor(y), Tensor(z))
    out = net(Tensor(x), Tensor(y), Tensor(z))
    assert np.allclose(out.asnumpy(), expect_out, 0.001, 0.001)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_custom_mul_aclnn_add_prefix(context_mode):
    """
    Feature: Custom op testcase
    Description: test case for adding prefix
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context_mode, save_graphs=False, save_graphs_path="./graphs",
                        jit_config={"jit_level": "O0"})
    x = np.ones([8, 2048]).astype(np.float16)
    y = np.ones([8, 2048]).astype(np.float16)
    z = np.random.rand(8, 2048).astype(np.float16)
    net = CustomNetAddPrefix("Mul", lambda x, _: x, None)
    expect_out = (x + y) * y - z
    out = net(Tensor(x), Tensor(y), Tensor(z))
    assert np.allclose(out.asnumpy(), expect_out, 0.001, 0.001)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_custom_mul_aclnn_infer_cpp(context_mode):
    """
    Feature: Custom op testcase
    Description: test case for inferring shape by cpp
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context_mode, save_graphs=False, save_graphs_path="./graphs",
                        jit_config={"jit_level": "O0"})
    x = np.ones([8, 2048]).astype(np.float16)
    y = np.ones([8, 2048]).astype(np.float16)
    z = np.random.rand(8, 2048).astype(np.float16)
    net = CustomNetAddPrefix("./infer_file/custom_cpp_infer.cc:Mul", None, None)
    expect_out = (x + y) * y - z
    out = net(Tensor(x), Tensor(y), Tensor(z))
    assert np.allclose(out.asnumpy(), expect_out, 0.001, 0.001)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_custom_mul_aclnn_bprop(context_mode):
    """
    Feature: Custom op testcase
    Description: test case for custom mul backpropagation.
    Expectation: the result match with numpy result
    """

    def bprop(x, y, out, dout):
        return dout, dout

    context.set_context(mode=context_mode, save_graphs=False, save_graphs_path="./graphs",
                        jit_config={"jit_level": "O0"})
    x = np.ones([8, 2048]).astype(np.float16)
    y = np.ones([8, 2048]).astype(np.float16)
    z = np.random.rand(8, 2048).astype(np.float16)
    net = CustomNetAddPrefix("Mul", lambda x, _: x, bprop)
    base_net = BaseNet()
    dx = ops.GradOperation()(net)(Tensor(x), Tensor(y), Tensor(z))
    expect_dx = ops.GradOperation()(base_net)(Tensor(x), Tensor(y), Tensor(z))
    assert np.allclose(dx.asnumpy(), expect_dx.asnumpy(), 0.001, 0.001)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE])
def test_custom_mul_aclop(context_mode):
    """
    Feature: Custom op testcase
    Description: test case for Adding prefix
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context_mode, save_graphs=False, save_graphs_path="./graphs",
                        jit_config={"jit_level": "O0"})
    x = np.ones([8, 2048]).astype(np.float16)
    y = np.ones([8, 2048]).astype(np.float16)
    z = np.random.rand(8, 2048).astype(np.float16)
    net = CustomNetAclOp("Mul", lambda x, _: x, None)
    expect_out = (x + y) * y - z
    out = net(Tensor(x), Tensor(y), Tensor(z))
    assert np.allclose(out.asnumpy(), expect_out, 0.001, 0.001)
