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
import pytest

import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import context
from mindspore import jit
from mindspore.ops.auto_generate import apply_rotary_pos_emb_
from tests.st.utils import test_utils

def apply_rotary_pos_emb_exec(query, key, cos, sin):
    x1 = query[..., :64]
    x2 = query[..., 64:]
    concat = np.concatenate((-x2, x1), axis=-1)
    x2_mul = concat * sin
    x1_mul = query * cos
    res0 = x2_mul + x1_mul

    k1 = key[..., :64]
    k2 = key[..., 64:]
    concatk = np.concatenate((-k2, k1), axis=-1)
    x1k_mul = concatk * sin
    x2k_mul = key * cos
    res1 = x2k_mul + x1k_mul
    return res0, res1

@test_utils.run_with_cell
def moe_init_routing_forward_func(query, key, cos, sin, layout):
    batch_valid_length = ms.Tensor(np.ones((1, 1)), mstype.int32)
    return apply_rotary_pos_emb_(query, key, cos, sin, batch_valid_length, cos_format=layout)


@pytest.mark.parametrize('mode', ['GE'])
def test_apply_rotary_pos_emb_case0(mode):
    """
    Feature: Test the apply_rotary_pos_emb_ calculate
    Description: Test the moe_init_routing ops in Ascend backend
    Expectation: The result match to the expect value.
    """
    # numpy input
    query_data = np.random.uniform(0, 1, [4, 1024, 16, 128]).astype(np.float16)

    key_data = np.random.uniform(0, 1, [4, 1024, 16, 128]).astype(np.float16)

    cos_data = np.random.uniform(0, 1, [4, 1024, 1, 128]).astype(np.float16)

    sin_data = np.random.uniform(0, 1, [4, 1024, 1, 128]).astype(np.float16)

    query_exec, key_exec = apply_rotary_pos_emb_exec(query_data, key_data, cos_data, sin_data)
    # tensor input
    query = ms.Tensor(query_data, ms.float16)
    key = ms.Tensor(key_data, ms.float16)
    cos = ms.Tensor(cos_data, ms.float16)
    sin = ms.Tensor(sin_data, ms.float16)
    batch_valid_length = ms.Tensor(np.ones((1, 1)), mstype.int32)

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
        query_ms, key_ms = apply_rotary_pos_emb_(query, key, cos, sin, batch_valid_length, cos_format=1)
    elif mode == 'KBK':
        context.set_context(mode=ms.GRAPH_MODE)
        query_ms, key_ms = (jit(apply_rotary_pos_emb_, jit_level="O0"))\
            (query, key, cos, sin, batch_valid_length, cos_format=1)
    else:
        context.set_context(mode=ms.GRAPH_MODE)
        query_ms, key_ms = (jit(apply_rotary_pos_emb_, backend="GE"))\
            (query, key, cos, sin, batch_valid_length, cos_format=1)

    np.testing.assert_allclose(query_ms.asnumpy(), query_exec, rtol=1e-3)
    np.testing.assert_allclose(key_ms.asnumpy(), key_exec, rtol=1e-3)
