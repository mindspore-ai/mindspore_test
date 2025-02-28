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
from mindspore import dtype as mstype
from mindspore import context, Tensor, ops
from mindspore.nn import Cell
from mindspore.ops.auto_generate import grouped_matmul_v4, GroupedMatmulV4

from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.mark_utils import arg_mark


# GroupedMatmul has 8 inputs and 1 outputs
# -----------------Input-----------------
# 1.x:                         TensorList ((N, h), ) or ((bs, N, h), )
# 2.weight:                    TensorList ((h, 4h)...(h, 4h)) or ((E, h, 4h))
# optional input
# 3.bias:                      TensorList (empty_tensor,)
# 4.scale:                     TensorList (empty_tensor,)
# 5.offset:                    TensorList (empty_tensor,)
# 6.antiquant_scale:           TensorList (empty_tensor,)
# 7.antiquant_offset:          TensorList (empty_tensor,)
# 8.pre_token_scale:           TensorList (empty_tensor,)
# 9.group_list:                Tensor
# 10.activation_input:         TensorList (empty_tensor,)
# 11.activation_quant_scale:   TensorList (empty_tensor,)
# 12.activation_quant_offset:  TensorList (empty_tensor,)
# 13.split_item:               int(0/1/2/3, currently only support 0/3)
# 14.group_type:               int(-/0/1/2, currently only support -1/0)
# 15.group_list_type:          int(0/1)
# 16.act_type:                 int(0/1/2/3/4/5, currently not supported)
# ------------------------------
# y:                     TensorList ((N, 4h), ) or ((bs, N, 4h), )

def my_cmp(np1, np2, rtol=1e-3):
    print("np1.shape: ", np1.shape)
    print("np2.shape: ", np2.shape)
    print("max diff:  ", np.max(np1 - np2))
    diffidx = ~np.isclose(np1, np2, rtol=rtol)  # true is not close
    diffratio = np.around(diffidx.sum() / diffidx.size, 4)
    print("np1 diff num: ", np1[diffidx])
    print("np2 diff num: ", np2[diffidx])
    print("diff(", str(rtol), ") ratio: ", diffratio)


def get_empty_tensor(dtype=mstype.float32):
    x = Tensor([1], dtype)
    output = ops.slice(x, (0,), (0,))
    return output


def split_x(x, group_list):
    x_split = []
    for i in range(len(group_list)):
        if i == 0:
            x_split.append(x[0: group_list[i],])
        else:
            x_split.append(x[group_list[i - 1]: group_list[i],])
    return x_split


def split_w(w):
    tmp_split = np.split(w, w.shape[0], axis=0)
    w_split = []
    for t in tmp_split:
        w_split.append(np.squeeze(t, 0))
    return w_split


class GroupedMatmulV4Net(Cell):
    def __init__(self):
        super().__init__()
        self.gmm_v4 = GroupedMatmulV4()

    def construct(self, x, weight, bias=None, scale=None, offset=None, antiquant_scale=None, antiquant_offset=None,
                  pertoken_scale=None, group_list=None, split_item=3, group_type=-1, group_list_type=0):
        out = self.gmm_v4(x, weight, bias, scale, offset, antiquant_scale, antiquant_offset, pertoken_scale, group_list,
                          split_item=split_item, group_type=group_type, group_list_type=group_list_type)
        return out


@test_utils.run_with_cell
def grouped_matmul_v4_forward_func(x, weight, group_list):
    out = grouped_matmul_v4([x,], [weight,], group_list=group_list, split_item=3, group_type=0, group_list_type=1)
    return out[0]


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_grouped_matmul_v4_x2d_w3d_splititem3_grouptype0_none_a8w8():
    """
    Feature: Test grouped_matmul
    Description: semi_auto_parallel
    Expectation: shape is as expected.
    """
    context.set_context(device_target="Ascend")
    ms.set_context(mode=ms.GRAPH_MODE, jit_level='O0')
    gmm_v4_net = GroupedMatmulV4Net()

    split_item = 3
    group_type = 0
    group_list_type = 1

    M0 = 32
    K0 = 256
    N0 = 128
    E0 = 8
    group_list_np = [1, 2, 7, 4, 4, 4, 2, 8]

    # numpy calculate
    np_x_all = np.random.uniform(-128, 127, size=[M0, K0]).astype(np.int8)
    np_w_all = np.random.uniform(-128, 127, size=[E0, K0, N0]).astype(np.int8)
    np_s_all = np.array(np.full([E0, N0], 0.1)).astype(np.float32)
    np_pts = np.array([0.1] * M0).astype(np.float32)

    np_x = split_x(np_x_all, np.cumsum(group_list_np))
    np_w = split_w(np_w_all)
    np_s = split_w(np_s_all)
    res_np = [np.matmul(x0, w0 * s0) for x0, w0, s0 in zip(np_x, np_w, np_s)]
    except_np = np.concatenate(res_np, axis=0) * np_pts.reshape(M0, 1)

    # ms calculate
    x = [ms.Tensor(np_x_all)]
    w = [ms.Tensor(np_w_all)]
    scale = [ms.Tensor(np_s_all, dtype=mstype.bfloat16)]
    pertoken_scale = [ms.Tensor(np_pts)]

    b = None
    offset = None
    antiquant_scale = None
    antiquant_offset = None
    group_list = ms.Tensor(group_list_np, dtype=mstype.int64)

    res = gmm_v4_net(x, w, b, scale, offset, antiquant_scale, antiquant_offset, pertoken_scale, group_list,
                     split_item, group_type, group_list_type)

    # compare
    np.testing.assert_allclose(except_np, res[0].float().asnumpy(), rtol=5e-3)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
def test_grouped_matmul_v4__dyn_shape():
    """
    Feature: Ops
    Description: test op GroupedMatmul with gorup type 0
    Expectation: expect correct result.
    """
    context.set_context(runtime_num_threads=1)  # multi-threads have none-initialized bug now.

    m = 10
    k = 20
    n = 14
    group_list = Tensor([2, 4, 2, 2])
    x = Tensor(np.random.randn(m, k).astype(np.float32))
    w = Tensor(np.random.randn(group_list.shape[0], k, n).astype(np.float32))
    inputs_0 = [x, w, group_list]

    m = 20
    k = 30
    n = 8
    group_list = Tensor([2, 4, 2, 2, 4, 3, 3])
    x = Tensor(np.random.randn(m, k).astype(np.float32))
    w = Tensor(np.random.randn(group_list.shape[0], k, n).astype(np.float32))
    inputs_1 = [x, w, group_list]

    TEST_OP(
        grouped_matmul_v4_forward_func,
        [
            inputs_0,
            inputs_1,
        ],
        "",
        disable_input_check=True,
        disable_grad=True,
        disable_yaml_check=True,
        disable_mode=['GRAPH_MODE',]
    )
