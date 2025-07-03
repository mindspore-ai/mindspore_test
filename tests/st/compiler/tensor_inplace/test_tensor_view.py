# Copyright 2024-2025 Huawei Technologies Co., Ltd
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
# ==============================================================================
import numpy as np
import mindspore as ms
from mindspore import Tensor
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_max():
    """
    Feature: Support view ops with keyword args input.
    Description: Support view ops with keyword args input.
    Expectation: Run success.
    """
    def tensor_max(tensor, axis=None, keepdims=False, initial=None, where=True):
        return tensor.max(axis, keepdims, initial=initial, where=where)

    @ms.jit(backend="ms_backend")
    def func():
        a = np.random.rand(2, 3).astype(np.float32)
        b = Tensor(a)
        where = np.random.randint(low=0, high=2,
                                  size=[2, 3]).astype(np.bool_)
        out_np = a.max(initial=2.0, where=where, axis=-1)
        out_ms = tensor_max(Tensor(b), initial=2.0,
                            where=Tensor(where), axis=-1)
        return out_np, out_ms.asnumpy()
    np_array, ms_array = func()
    np.allclose(np_array, ms_array, rtol=5e-03, atol=1.e-8)
