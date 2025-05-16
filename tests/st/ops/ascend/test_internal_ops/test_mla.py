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

"""test mla"""

import mindspore as ms
from mindspore import nn, ops, Tensor, context
from mindspore.ops.operations.nn_ops import PagedAttention
import numpy as np
import pytest


class MlaTestParam:
    """MlaTestParam"""
    def __init__(self, num_heads, kv_heads, block_size, head_size_nope, head_size_rope, num_blocks,
        q_seq_lens : list, context_lengths : list, tor, nope_ms_dtype, rope_ms_dtype, mask_mode : str):

        self.num_heads = num_heads
        self.kv_heads = kv_heads
        self.block_size = block_size
        self.head_size_nope = head_size_nope
        self.head_size_rope = head_size_rope
        self.num_blocks = num_blocks
        self.q_seq_lens = q_seq_lens
        self.context_lengths = context_lengths
        self.tor = tor
        self.nope_ms_dtype = nope_ms_dtype
        self.rope_ms_dtype = rope_ms_dtype
        self.mask_mode = mask_mode
        self.mask_factor = -10000.0 if rope_ms_dtype == ms.float16 else 1.0

        self.batch = len(q_seq_lens)

        self.max_context_len = max(context_lengths)
        self.max_num_blocks_per_seq = (self.max_context_len + block_size - 1) // block_size

        self.num_tokens = (int)(np.array(q_seq_lens).sum())
        self.block_tables = self._build_block_tables()

        self._build_tensor_inputs()


    def _build_np_mask(self):
        """_build_np_mask"""
        if self.mask_mode == "MASK_NONE":
            return None

        if self.mask_mode == "MASK_SPEC":
            pre_qseqlen = 0
            np_mask = np.zeros(shape=(self.num_tokens, self.max_context_len)).astype(np.float32)
            for i in range(self.batch):
                qseqlen = self.q_seq_lens[i]
                kseqlen = self.context_lengths[i]
                tri = np.ones((qseqlen, qseqlen))
                tri = np.triu(tri, 1)
                tri *= self.mask_factor
                np_mask[pre_qseqlen:(pre_qseqlen + qseqlen), kseqlen-qseqlen:kseqlen] = tri
                pre_qseqlen += qseqlen
            return np_mask

        if self.mask_mode == "MASK_FREE":
            pass

        return None


    def _build_block_tables(self):
        """_build_block_tables"""
        block_tables_list = []
        for i in range(self.num_tokens):
            block_table = [
                i * self.max_num_blocks_per_seq + _ for _ in range(self.max_num_blocks_per_seq)
            ]
            block_tables_list.append(block_table)
        return block_tables_list


    def _build_tensor_inputs(self):
        """_build_tensor_inputs"""
        np_q_nope = np.random.uniform(-1.0, 1.0, size=(self.num_tokens, self.num_heads, self.head_size_nope))
        np_q_rope = np.random.uniform(-1.0, 1.0, size=(self.num_tokens, self.num_heads, self.head_size_rope))
        np_ctkv = np.random.uniform(-1.0, 1.0, size=(self.num_blocks, self.block_size,
                                                     self.kv_heads, self.head_size_nope))
        np_k_rope = np.random.uniform(-1.0, 1.0, size=(self.num_blocks, self.block_size,
                                                       self.kv_heads, self.head_size_rope))

        np_context_lens = np.array(self.context_lengths).astype(np.int32)
        np_q_seq_lens = np.array(self.q_seq_lens).astype(np.int32)

        self.q_nope_tensor = Tensor(np_q_nope, dtype=self.nope_ms_dtype)
        self.q_rope_tensor = Tensor(np_q_rope, dtype=self.rope_ms_dtype)
        self.ctkv_tensor = Tensor(np_ctkv, dtype=self.nope_ms_dtype)
        self.k_rope_tensor = Tensor(np_k_rope, dtype=self.rope_ms_dtype)

        self.block_tables_tensor = Tensor(np.array(self.block_tables).astype(np.int32))

        np_mask = self._build_np_mask()
        self.mask_tensor = None if np_mask is None else Tensor(np_mask, dtype=self.rope_ms_dtype)

        if self.nope_ms_dtype == ms.int8:
            self.deq_scale_qk_tensor = Tensor(np.random.uniform(-1.0, 1.0, size=(self.num_heads, )), dtype=ms.float32)
            self.deq_scale_pv_tensor = Tensor(np.random.uniform(-1.0, 1.0, size=(self.num_heads, )), dtype=ms.float32)
        else:
            self.deq_scale_qk_tensor = None
            self.deq_scale_pv_tensor = None

        self.q_seq_lens_tensor = Tensor(np_q_seq_lens)
        self.context_lengths_tensor = Tensor(np_context_lens)


class Net(nn.Cell):
    """Net"""
    def __init__(self, q_head_num, kv_head_num, mask_type, tor):
        super().__init__()
        self.q_head_num = q_head_num
        self.kv_head_num = kv_head_num
        self.mask_type = mask_type
        self.tor = tor

    def construct(self, q_nope, q_rope, ctkv, k_rope, block_tables, mask, deq_scale_qk, deq_scale_pv,
                  q_seq_lens, batch_valid_length):
        return ops.auto_generate.mla(q_nope, q_rope, ctkv, k_rope, block_tables, mask, deq_scale_qk,
                                     deq_scale_pv, q_seq_lens, batch_valid_length, self.q_head_num, self.tor,
                                     self.kv_head_num, self.mask_type)


class GoldenNet(nn.Cell):
    """GoldenNet"""
    def __init__(self, q_head_num, kv_head_num, mask_mode, tor, mla_v_dim):
        super().__init__()
        self.q_head_num = q_head_num
        self.kv_head_num = kv_head_num
        self.mask_mode = mask_mode
        self.tor = tor
        self.mla_v_dim = mla_v_dim
        self.op = PagedAttention(self.q_head_num, self.tor, self.kv_head_num, 'DEFAULT', 'MASK_DEFAULT',
                                 self.mla_v_dim)

    def construct(self, query, key_cache, value_cache, block_tables, batch_valid_length, antiquant_scale,
                  antiquant_offset, attn_mask, q_seq_lens, alibi_mask):
        return self.op(query, key_cache, value_cache, block_tables, batch_valid_length, antiquant_scale,
                       antiquant_offset, attn_mask, q_seq_lens, alibi_mask)


def run_mla(test_param : MlaTestParam):
    """run mla"""
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    dyn_q_nope_shape = [None for _ in test_param.q_nope_tensor.shape]
    dyn_q_nope_tensor = Tensor(shape=dyn_q_nope_shape, dtype=test_param.q_nope_tensor.dtype)

    net = Net(test_param.num_heads, test_param.kv_heads, test_param.mask_mode, test_param.tor)
    net.set_inputs(q_nope=dyn_q_nope_tensor)

    out, _ = net(test_param.q_nope_tensor, test_param.q_rope_tensor, test_param.ctkv_tensor, test_param.k_rope_tensor,
                 test_param.block_tables_tensor, test_param.mask_tensor, test_param.deq_scale_qk_tensor,
                 test_param.deq_scale_pv_tensor, test_param.q_seq_lens_tensor, test_param.context_lengths_tensor)
    return out


def run_golden(test_param : MlaTestParam):
    """run_golden"""
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})

    mla_v_dim = 512
    query = ops.reshape(ops.concat((test_param.q_nope_tensor, test_param.q_rope_tensor), axis=-1),
                        (test_param.num_tokens, 1, -1))
    key_cache = ops.concat((test_param.ctkv_tensor, test_param.k_rope_tensor), axis=-1)
    dyn_q_shape = [None for _ in test_param.q_nope_tensor.shape]
    dyn_q_nope_tensor = Tensor(shape=dyn_q_shape, dtype=test_param.q_nope_tensor.dtype)
    golden_net = GoldenNet(test_param.num_heads, test_param.kv_heads, "MASK_DEFAULT", test_param.tor, mla_v_dim)
    golden_net.set_inputs(query=dyn_q_nope_tensor)

    out_golden = golden_net(query, key_cache, key_cache, test_param.block_tables_tensor,
                            test_param.context_lengths_tensor, None, None, test_param.mask_tensor,
                            test_param.q_seq_lens_tensor, None)

    return out_golden


def run_test(test_param : MlaTestParam):
    """run test"""
    out_actual = run_mla(test_param)
    out_golden = run_golden(test_param)

    assert np.allclose(out_actual.astype(ms.float32).asnumpy().reshape(-1),
                       out_golden.astype(ms.float32).asnumpy().reshape(-1), 0.001, 0.001)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [ms.float16, ms.bfloat16])
@pytest.mark.parametrize('mask_mode', ["MASK_NONE", "MASK_SPEC"])
def test_mla_base(dtype, mask_mode):
    """
    Feature: test mla
    Description: test mla.
    Expectation: the result is correct
    """
    q_seq_lens = [1, 1, 1, 1]
    context_lengths = [192, 193, 194, 195]
    test_param = MlaTestParam(32, 1, 128, 512, 64, 1024, q_seq_lens, context_lengths, 0.001, dtype, dtype, mask_mode)
    run_test(test_param)


# @pytest.mark.level0
# @pytest.mark.platform_arm_ascend910b_training
# @pytest.mark.env_onecard
# @pytest.mark.parametrize('mask_mode', ["MASK_NONE", "MASK_SPEC"])
# def test_mla_int8(mask_mode):
#     """
#     Feature: test mla
#     Description: test mla.
#     Expectation: the result is correct
#     """
#     q_seq_lens = [1, 1, 1, 1]
#     context_lengths = [192, 193, 194, 195]
#     test_param = MlaTestParam(32, 1, 128, 512, 64, 1024, q_seq_lens, context_lengths, 0.001, ms.int8, ms.bfloat16, mask_mode)
#     run_test(test_param)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('block_size', [16, 32, 64, 128])
@pytest.mark.parametrize('mask_mode', ["MASK_NONE", "MASK_SPEC"])
def test_mla_block_size(block_size, mask_mode):
    """
    Feature: test mla
    Description: test mla.
    Expectation: the result is correct
    """
    q_seq_lens = [1, 1, 1, 1]
    context_lengths = [192, 193, 194, 195]
    test_param = MlaTestParam(32, 1, block_size, 512, 64, 1024, q_seq_lens, context_lengths,
                              0.001, ms.float16, ms.float16, mask_mode)
    run_test(test_param)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [ms.bfloat16, ms.float16])
@pytest.mark.parametrize('mask_mode', ["MASK_NONE"])
@pytest.mark.parametrize('block_size', [16, 128])
def test_mla_mtp(dtype, mask_mode, block_size):
    """
    Feature: test mla
    Description: test mla.
    Expectation: the result is correct
    """
    q_seq_lens = [1, 1, 2, 1]
    context_lengths = [192, 193, 194, 195]
    test_param = MlaTestParam(4, 1, block_size, 512, 64, 128, q_seq_lens, context_lengths,
                              0.001, dtype, dtype, mask_mode)
    run_test(test_param)
