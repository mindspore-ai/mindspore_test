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

import numpy as np
import mindspore as ms
from mindspore import mint, ops
from mindspore import nn, Tensor, Parameter, context
from mindspore.ops import _add_attr
from parallel.utils.utils import ParallelValidator, compile_net
from tests.ut.python.ops.test_math_ops import VirtualLoss


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, *x):
        predict = self.network(*x)
        return self.loss(predict)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network
        self.grad_all = ops.GradOperation(get_all=True)
    def construct(self, *x):
        return self.grad_all(self.network)(*x)

class TestFunctionalAddPrimAttrs:
    def setup_method(self):
        pass

    def teardown_method(self):
        pass

    def _get_nodes_info(self, graph_validator):
        d = graph_validator.graph_info_dict
        node_infos = []
        for _, nodes in d.items():
            for node, node_info in nodes.items():
                node_infos.append((node, node_info))
        return node_infos

    def _check_node_and_primattr(self, node_infos, expect_node_name, expect_primattr_dict):
        for node_info in node_infos:
            node_name, node_prim_attr = node_info[0], node_info[1]['attrs']
            if node_name.startswith(expect_node_name) and \
                set(expect_primattr_dict.items()).issubset(node_prim_attr.items()):
                return True
        return False

    def _check_no_addattr_node_in_graph(self, node_infos):
        for node_info in node_infos:
            node_name, _ = node_info[0], node_info[1]['attrs']
            if node_name.startswith('AddAttr'):
                return False
        return True

    def test_addattr_for_functional_matmul(self):
        """
        Feature: test _add_attr for functional mint
        Description:  add primitive attr for functional mint
        Expectation: assert pass
        """
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", dataset_strategy="full_batch")
        context.set_auto_parallel_context(device_num=8, global_rank=0)
        class MatMulCell(nn.Cell):
            def __init__(self):
                super(MatMulCell, self).__init__()
                self.w1 = Parameter(Tensor(np.random.randn(32, 128).astype(np.float16)), name='weight1')
                self.tagged_mm = _add_attr(mint.matmul, test1=123, test2=True)
            def construct(self, x):
                out = self.tagged_mm(x, self.w1)
                return out

        input_x = Tensor(np.random.randn(16, 32), dtype=ms.float16)
        net = GradWrap(NetWithLoss(MatMulCell()))
        phase = compile_net(net, input_x)
        validator = ParallelValidator(net, phase)
        node_infos = self._get_nodes_info(validator)
        assert self._check_node_and_primattr(node_infos, 'MatMulExt', {'test1': 123, 'test2': True}) and \
               self._check_no_addattr_node_in_graph(node_infos)

    def test_use_addattr_for_multiple_operator(self):
        """
        Feature: test _add_attr for multiple operators
        Description: _add_attr for multiple operators and nested _add_attr
        Expectation: assert pass
        """
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", dataset_strategy="full_batch")
        context.set_auto_parallel_context(device_num=8, global_rank=0)
        class MatMulCell(nn.Cell):
            def __init__(self):
                super(MatMulCell, self).__init__()
                self.w1 = Parameter(Tensor(np.random.randn(32, 128).astype(np.float16)), name='weight1')
                self.tagged_mm = _add_attr(_add_attr(mint.matmul, test1=123, test2=True), embbed=True)
                self.tagged_relu = _add_attr(mint.nn.functional.relu, relu_attr=1)

            def construct(self, x):
                out = self.tagged_mm(x, self.w1)
                out = self.tagged_relu(out)
                return out

        input_x = Tensor(np.random.randn(16, 32), dtype=ms.float16)
        net = GradWrap(NetWithLoss(MatMulCell()))
        phase = compile_net(net, input_x)
        validator = ParallelValidator(net, phase)
        node_infos = self._get_nodes_info(validator)
        assert self._check_node_and_primattr(node_infos, 'MatMulExt', \
                                    {'test1': 123, 'test2': True, 'embbed': True}) and \
               self._check_node_and_primattr(node_infos, 'ReLU', {'relu_attr': 1}) and \
               self._check_no_addattr_node_in_graph(node_infos)

    def test_use_addattr_in_standalone_mode(self):
        """
        Feature: test _add_attr for multiple operators in standalone mode
        Description: _add_attr for multiple operators and nested _add_attr
        Expectation: compile pass
        """
        context.set_auto_parallel_context(parallel_mode="stand_alone")
        class MatMulCell(nn.Cell):
            def __init__(self):
                super(MatMulCell, self).__init__()
                self.w1 = Parameter(Tensor(np.random.randn(32, 128).astype(np.float16)), name='weight1')
                self.tagged_mm = _add_attr(_add_attr(mint.matmul, test1=123, test2=True), embbed=True)
                self.tagged_relu = _add_attr(mint.nn.functional.relu, relu_attr=1)

            def construct(self, x):
                out = self.tagged_mm(x, self.w1)
                out = self.tagged_relu(out)
                return out

        input_x = Tensor(np.random.randn(16, 32), dtype=ms.float16)
        net = GradWrap(NetWithLoss(MatMulCell()))
        compile_net(net, input_x)
