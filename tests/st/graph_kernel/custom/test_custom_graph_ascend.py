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
""" tests_custom_pyboost_ascend """

import numpy as np
import mindspore as ms
from mindspore.ops import CustomOpBuilder
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_custom_single_operator():
    """
    Feature: Single-operator loading via CustomOpBuilder.
    Description: Build and execute the add4 operator defined in YAML and CPP files in GRAPH mode.
    Expectation: Execution succeeds and outputs match NumPy results.
    """
    ms.set_device("Ascend")
    ms.set_context(mode=ms.GRAPH_MODE, save_graphs=False, save_graphs_path="./graphs")

    class MyNet(ms.nn.Cell):
        def __init__(self):
            super(MyNet, self).__init__()
            self.my_ops = CustomOpBuilder("graphmode_add",
                                          ["jit_test_files/graph/add4.cpp",
                                           "jit_test_files/graph/module.cpp"],
                                          backend="Ascend", op_def=["jit_test_files/graph/add4.yaml"],
                                          op_doc=["jit_test_files/graph/add4_doc.yaml"]).load()

        def construct(self, x, y):
            return self.my_ops.add4(x, y, 1)

    x = np.array([1, 2, 3], dtype=np.float16)
    y = np.array([4, 5, 6], dtype=np.float16)
    output = MyNet()(ms.Tensor(x), ms.Tensor(y))
    expect = x + y
    print(output.asnumpy())
    assert np.allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_custom_single_operator_no_doc():
    """
    Feature: Single-operator loading without documentation YAML.
    Description: Build and execute the add4 operator defined only by operator YAML and CPP files.
    Expectation: Execution succeeds and outputs match NumPy results.
    """
    ms.set_device("Ascend")
    ms.set_context(mode=ms.GRAPH_MODE, save_graphs=False, save_graphs_path="./graphs")

    class MyNet(ms.nn.Cell):
        def __init__(self):
            super(MyNet, self).__init__()
            self.my_ops = CustomOpBuilder("graphmode_add",
                                          ["jit_test_files/graph/add4.cpp",
                                           "jit_test_files/graph/module.cpp"],
                                          backend="Ascend", op_def=["jit_test_files/graph/add4.yaml"]).load()

        def construct(self, x, y):
            return self.my_ops.add4(x, y, 1)

    x = np.array([1, 2, 3], dtype=np.float16)
    y = np.array([4, 5, 6], dtype=np.float16)
    output = MyNet()(ms.Tensor(x), ms.Tensor(y))
    expect = x + y
    print(output.asnumpy())
    assert np.allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_custom_single_operator_func():
    """
    Feature: Single-operator execution from a JIT-compiled function.
    Description: Load add4 operator via CustomOpBuilder and call it inside a JIT function.
    Expectation: Execution succeeds and outputs match NumPy results.
    """
    ms.set_device("Ascend")
    ms.set_context(mode=ms.GRAPH_MODE, save_graphs=False, save_graphs_path="./graphs")

    my_ops = CustomOpBuilder("graphmode_add",
                             ["jit_test_files/graph/add4.cpp",
                              "jit_test_files/graph/module.cpp"],
                             backend="Ascend", op_def=["jit_test_files/graph/add4.yaml"],
                             op_doc=["jit_test_files/graph/add4_doc.yaml"]).load()

    @ms.jit()
    def add_net(x, y):
        return my_ops.add4(x, y)

    x = np.array([1, 2, 3], dtype=np.float16)
    y = np.array([4, 5, 6], dtype=np.float16)
    output = add_net(ms.Tensor(x), ms.Tensor(y))
    expect = x + y
    print(output.asnumpy())
    assert np.allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_custom_multi_operator():
    """
    Feature: Multiple-operator loading and chaining.
    Description: Build both add4 and add3 operators, chain them in a single graph, and verify cumulative results.
    Expectation: Execution succeeds and final outputs equal x + y + y.
    """
    ms.set_device("Ascend")
    ms.set_context(mode=ms.GRAPH_MODE, save_graphs=False, save_graphs_path="./graphs")

    class MyNet(ms.nn.Cell):
        def __init__(self):
            super(MyNet, self).__init__()
            self.my_ops = CustomOpBuilder("graphmode_add_2",
                                          ["jit_test_files/graph/add4.cpp", "jit_test_files/graph/add3.cpp",
                                           "jit_test_files/graph/module.cpp"], backend="Ascend",
                                          op_def=["jit_test_files/graph/add4.yaml",
                                                  "jit_test_files/graph/add3.yaml"]).load()

        def construct(self, x, y):
            out = self.my_ops.add4(x, y, 1)
            return self.my_ops.add3(out, y, 1)

    x = np.array([1, 2, 3], dtype=np.float16)
    y = np.array([4, 5, 6], dtype=np.float16)
    output = MyNet()(ms.Tensor(x), ms.Tensor(y))
    expect = x + y + y
    print(output.asnumpy())
    assert np.allclose(output.asnumpy(), expect)


def test_graphmode_add_offline():
    """
    Feature: Offline-compiled custom operator.
    Description: Use a pre-built add4 operator imported as a Python module inside a JIT graph.
    Expectation: Execution succeeds and outputs equal x + y.
    """
    ms.set_device("Ascend")
    ms.set_context(mode=ms.GRAPH_MODE, save_graphs=False, save_graphs_path="./graphs")
    import custom_ops

    @ms.jit()
    def add_net(x, y):
        out = custom_ops.add4(x, y)
        return custom_ops.add4(out, y)

    x = np.array([1, 2, 3], dtype=np.float16)
    y = np.array([4, 5, 6], dtype=np.float16)
    output = add_net(ms.Tensor(x), ms.Tensor(y))
    expect = x + y + y
    assert np.allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_CustomOpBuilder_exception_1():
    """
    Feature: CustomOpBuilder error handling in PyNative mode.
    Description: Attempt to run a custom operator built for GRAPH mode under PyNative mode.
    Expectation: RuntimeError containing 'not support PyNative mode' is raised.
    """
    ms.set_device("Ascend")
    ms.set_context(mode=ms.PYNATIVE_MODE, save_graphs=False, save_graphs_path="./graphs")

    class MyNet(ms.nn.Cell):
        def __init__(self):
            super(MyNet, self).__init__()
            self.my_ops = CustomOpBuilder("graphmode_add",
                                          ["jit_test_files/graph/graphmode_add4.cpp",
                                           'jit_test_files/pyboost_aclnn_sum.cpp',
                                           "jit_test_files/graph/module.cpp"],
                                          backend="Ascend", op_def=["jit_test_files/graph/add4.yaml"],
                                          op_doc=["jit_test_files/graph/add4_doc.yaml"]).load()

        def construct(self, x, y):
            return self.my_ops.add4(x, y, 1)

    x = np.array([1, 2, 3], dtype=np.float16)
    y = np.array([4, 5, 6], dtype=np.float16)
    try:
        MyNet()(ms.Tensor(x), ms.Tensor(y))
    except RuntimeError as e:
        assert "not support PyNative mode" in str(e)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_CustomOpBuilder_exception_2():
    """
    Feature: CustomOpBuilder unsupported GRAPH mode operator.
    Description: Attempt to invoke an operator that explicitly does not support GRAPH mode.
    Expectation: AttributeError containing 'does not support GRAPH mode' is raised.
    """
    ms.set_device("Ascend")
    ms.set_context(mode=ms.GRAPH_MODE, save_graphs=False, save_graphs_path="./graphs")

    class MyNet(ms.nn.Cell):
        def __init__(self):
            super(MyNet, self).__init__()
            self.my_ops = CustomOpBuilder("graphmode_add",
                                          ["jit_test_files/graph/graphmode_add4.cpp",
                                           'jit_test_files/pyboost_aclnn_sum.cpp',
                                           "jit_test_files/graph/module.cpp"],
                                          backend="Ascend", op_def=["jit_test_files/graph/add4.yaml"],
                                          op_doc=["jit_test_files/graph/add4_doc.yaml"]).load()

        def construct(self, x, y, z):
            return self.my_ops.npu_abs_reduce_sum(x, y, z)

    x = np.random.rand(4, 5, 6).astype(np.float32)
    try:
        MyNet()(ms.Tensor(x), (1,), True)
    except AttributeError as e:
        assert "does not support GRAPH mode" in str(e)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_CustomOpBuilder_exception_3():
    """
    Feature: CustomOpBuilder missing operator lookup.
    Description: Attempt to call an operator not present in either function or shared-object modules.
    Expectation: AttributeError containing 'neither in func_module nor in so_module' is raised.
    """
    ms.set_device("Ascend")
    ms.set_context(mode=ms.PYNATIVE_MODE, save_graphs=False, save_graphs_path="./graphs")

    class MyNet(ms.nn.Cell):
        def __init__(self):
            super(MyNet, self).__init__()
            self.my_ops = CustomOpBuilder("graphmode_add",
                                          ["jit_test_files/graph/graphmode_add4.cpp",
                                           'jit_test_files/pyboost_aclnn_sum.cpp',
                                           "jit_test_files/graph/module.cpp"],
                                          backend="Ascend", op_def=["jit_test_files/graph/add4.yaml"],
                                          op_doc=["jit_test_files/graph/add4_doc.yaml"]).load()

        def construct(self, x, y, z):
            return self.my_ops.add3(x, y, z)

    x = np.random.rand(4, 5, 6).astype(np.float32)
    try:
        MyNet()(ms.Tensor(x), (1,), True)
    except AttributeError as e:
        assert "neither in func_module nor in so_module" in str(e)


if __name__ == "__main__":
    test_custom_single_operator()
