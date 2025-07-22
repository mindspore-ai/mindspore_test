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
from mindspore.ops import CustomOpBuilder, ModuleWrapper


def test_graphmode_add():
    """
    Feature: CustomOpBuilder basic usage.
    Description: Add two tensors with custom ops in graph mode.
    Expectation: success.
    """
    ms.set_device("Ascend")
    ms.set_context(mode=ms.GRAPH_MODE, save_graphs=True, save_graphs_path="./graphs")

    class MyNet(ms.nn.Cell):
        def __init__(self):
            super(MyNet, self).__init__()
            self.my_ops = CustomOpBuilder("graphmode_add",
                                          ["jit_test_files/graphmode_add2.cpp", "jit_test_files/graphmode_add3.cpp",
                                           "jit_test_files/module.cpp"], backend="Ascend").load()

        def construct(self, x, y):
            out = self.my_ops.add2(x, y, 1)
            return self.my_ops.add3(out, y, 1)

    x = np.array([1, 2, 3], dtype=np.float16)
    y = np.array([4, 5, 6], dtype=np.float16)
    output = MyNet()(ms.Tensor(x), ms.Tensor(y))
    print(output)


def test_graphmode_add_monad():
    """
    Feature: Side-effect memory attribute.
    Description: Add op flagged with side_effect_mem.
    Expectation: success.
    """
    ms.set_device("Ascend")
    ms.set_context(mode=ms.GRAPH_MODE, save_graphs=True, save_graphs_path="./graphs")

    class MyNet(ms.nn.Cell):
        def __init__(self):
            super(MyNet, self).__init__()
            self.my_ops = CustomOpBuilder("graphmode_add",
                                          ["jit_test_files/graphmode_add2.cpp", "jit_test_files/graphmode_add3.cpp",
                                           "jit_test_files/module.cpp"], backend="Ascend").load()
            self.add2 = self.my_ops.add2
            self.add2.add_prim_attr("side_effect_mem", True)
            self.add3 = self.my_ops.add3

        def construct(self, x, y):
            self.add2(x, y, 1)
            return self.my_ops.add3(x, y, 1)

    x = np.array([1, 2, 3], dtype=np.float16)
    y = np.array([4, 5, 6], dtype=np.float16)
    output = MyNet()(ms.Tensor(x), ms.Tensor(y))
    print(output)


def test_graphmode_add_import():
    """
    Feature: ModuleWrapper import.
    Description: Reuse prebuilt custom module.
    Expectation: success.
    """
    ms.set_device("Ascend")
    ms.set_context(mode=ms.GRAPH_MODE, save_graphs=True, save_graphs_path="./graphs")

    class MyNet(ms.nn.Cell):
        def __init__(self):
            super(MyNet, self).__init__()
            import graphmode_add
            self.custom_mod = ModuleWrapper("graphmode_add", graphmode_add)

        def construct(self, x, y):
            out = self.custom_mod.add2(x, y, 1)
            return self.custom_mod.add3(out, y, 1)

    x = np.array([1, 2, 3], dtype=np.float16)
    y = np.array([4, 5, 6], dtype=np.float16)
    output = MyNet()(ms.Tensor(x), ms.Tensor(y))
    print(output)


def test_graphmode_add_import_func():
    """
    Feature: JIT with imported op.
    Description: Custom add called in @jit.
    Expectation: success.
    """
    ms.set_device("Ascend")
    ms.set_context(mode=ms.GRAPH_MODE, save_graphs=True, save_graphs_path="./graphs")
    import graphmode_add
    def func_add(x, y):
        custom_mod = ModuleWrapper("graphmode_add", graphmode_add)
        return custom_mod.add2(x, y, 1)

    @ms.jit()
    def add_net(x, y):
        out = func_add(x, y)
        return func_add(out, y)

    x = np.array([1, 2, 3], dtype=np.float16)
    y = np.array([4, 5, 6], dtype=np.float16)
    output = add_net(ms.Tensor(x), ms.Tensor(y))
    print(output)


def test_graphmode_add_op_def():
    """
    Feature: YAML-defined operator.
    Description: Add4 loaded from YAML spec.
    Expectation: success.
    """
    ms.set_device("Ascend")
    ms.set_context(mode=ms.GRAPH_MODE, save_graphs=True, save_graphs_path="./graphs")

    class MyNet(ms.nn.Cell):
        def __init__(self):
            super(MyNet, self).__init__()
            self.my_ops = CustomOpBuilder("graphmode_add",
                                          ["jit_test_files/graphmode_add4.cpp",
                                           "jit_test_files/module.cpp"],
                                          backend="Ascend", op_def=["jit_test_files/add4.yaml"],
                                          op_doc=["jit_test_files/add4_doc.yaml"]).load()

        def construct(self, x, y):
            return self.my_ops.add4(x, y, 1)

    x = np.array([1, 2, 3], dtype=np.float16)
    y = np.array([4, 5, 6], dtype=np.float16)
    output = MyNet()(ms.Tensor(x), ms.Tensor(y))
    print(output)


def test_graphmode_add_offline():
    """
    Feature: Offline compiled custom op.
    Description: Prebuilt add4 used in graph.
    Expectation: success.
    """
    ms.set_device("Ascend")
    ms.set_context(mode=ms.GRAPH_MODE, save_graphs=True, save_graphs_path="./graphs")
    import custom_ops

    @ms.jit()
    def add_net(x, y):
        out = custom_ops.add4(x, y)
        return custom_ops.add4(out, y)

    x = np.array([1, 2, 3], dtype=np.float16)
    y = np.array([4, 5, 6], dtype=np.float16)
    output = add_net(ms.Tensor(x), ms.Tensor(y))
    print(output)


# test_graphmode_add()
# test_graphmode_add_import()
# test_graphmode_add_import_func()
test_graphmode_add_op_def()
# test_graphmode_add_offline()
