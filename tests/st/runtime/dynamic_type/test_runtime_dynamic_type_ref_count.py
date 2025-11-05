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
"""
Test ref count for control flow.
"""
import mindspore as ms
from mindspore import Tensor
import numpy as np
from tests.st.backend.ms_backend.common.backend_graph import BackendGraph
from tests.mark_utils import arg_mark

@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_any_type_base():
    """
    Feature: Test any type.
    Description: Test single any type op.
    Expectation: Run success.
    """
    shape = (3, 4)
    root_graph = BackendGraph()
    root_para_1 = root_graph.add_parameter(ms.float32, shape)
    root_para_2 = root_graph.add_parameter(ms.float32, shape)
    maketuple = root_graph.add_cnode("MakeTuple", root_para_1, root_para_2)
    script_valuenode = root_graph.add_valuenode("x + y")
    key_valuenode = root_graph.add_valuenode(("x", "y"))
    pyexecute = root_graph.add_cnode("PyExecute", script_valuenode, key_valuenode, maketuple)
    root_graph.set_target(pyexecute, "CPU")
    root_graph.add_return(pyexecute)
    root_graph.infer()
    root_graph.compile()
    x = Tensor(np.ones(shape).astype(np.float32))
    y = Tensor(np.ones(shape).astype(np.float32))
    out = root_graph(x, y)
    print(out)
    assert np.allclose(out.asnumpy(), np.ones(shape).astype(np.float32) * 2)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_any_type_ref_count():
    """
    Feature: Test parameter output in any type graph.
    Description: The ref count of parameter should be fixed.
    Expectation: Run success.
    """
    shape = (3, 4)
    root_graph = BackendGraph()
    root_para_1 = root_graph.add_parameter(ms.float32, shape)
    root_para_2 = root_graph.add_parameter(ms.float32, shape)
    # kernel graph0
    add1 = root_graph.add_cnode("Add", root_para_1, root_para_2)
    # kernel graph1
    sub = root_graph.add_cnode("Sub", add1, root_para_2)
    script_valuenode = root_graph.add_valuenode("x + y")
    key_valuenode = root_graph.add_valuenode(("x", "y"))
    maketuple = root_graph.add_cnode("MakeTuple", root_para_1, sub)
    pyexecute = root_graph.add_cnode("PyExecute", script_valuenode, key_valuenode, maketuple)
    root_graph.set_target(pyexecute, "CPU")
    root_graph.set_target(sub, "CPU")
    # kernel graph2
    add2 = root_graph.add_cnode("Add", pyexecute, root_para_1)
    load = root_graph.add_cnode("Load", add1, add2)
    mul = root_graph.add_cnode("Mul", load, root_para_2)
    # kernel graph3
    div = root_graph.add_cnode("RealDiv", mul, root_para_2)
    root_graph.set_target(div, "CPU")
    # kernel graph4, input load is a parameter output of kernel graph 2.
    add3 = root_graph.add_cnode("Add", load, div)
    root_graph.add_return(add3)
    root_graph.infer()
    root_graph.compile()
    x = Tensor(np.ones(shape).astype(np.float32))
    y = Tensor(np.ones(shape).astype(np.float32))
    out = root_graph(x, y)
    print(out)
    assert np.allclose(out.asnumpy(), np.ones(shape).astype(np.float32) * 4)
