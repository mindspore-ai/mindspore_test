# Copyright 2024 Huawei Technocasties Co., Ltd
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
from mindspore import ops, jit, Tensor
from tests.mark_utils import arg_mark
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


def generate_ones_input(shape, dtype):
    return np.ones(shape).astype(dtype)


@test_utils.run_with_cell
def bernoulli_forward_func(x, p, generator=None):
    return x.bernoulli_(p, generator=generator)


@test_utils.run_with_cell
def bernoulli_forward_func_grad(x, p, generator=None):
    x = x * 1
    return x.bernoulli_(p, generator=generator)


@jit(backend="ms_backend")
def bernoulli_backward_func(x, p, generator=None):
    grad = ops.GradOperation(get_all=True)
    return grad(bernoulli_forward_func_grad)(x, p, generator)


def set_context_mode(mode):
    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == "kbk":
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    else:
        raise ValueError(f"Unsupported mode {mode}")


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("mode", ["pynative", "kbk"])
@pytest.mark.parametrize("p_mode", ["tensor", "float"])
@pytest.mark.skip(reason="Different versions of the run package "
                  "yield different results,and further investigation is needed to determine the cause.")
def test_bernoulli_normal(mode, p_mode):
    """
    Feature: pyboost function.
    Description: test function Tensor.bernoulli_ forward and backward.
    Expectation: expect correct result.
    """
    set_context_mode(mode)

    if p_mode == "tensor":
        p = Tensor(generate_ones_input((5, 5), np.float32)) * 0.3
        expect = np.array([[0.0, 0.0, 0.0, 0.0, 1.0],
                           [0.0, 0.0, 0.0, 1.0, 1.0],
                           [0.0, 0.0, 1.0, 1.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0, 1.0]]).astype(np.float32)
    else:
        p = 0.3
        expect = np.array([[0.0, 1.0, 1.0, 1.0, 0.0],
                           [0.0, 1.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 1.0],
                           [0.0, 0.0, 0.0, 1.0, 0.0],
                           [1.0, 1.0, 0.0, 0.0, 0.0]]).astype(np.float32)

    expect_grad = np.zeros((5, 5)).astype(np.float32)

    ms.manual_seed(10)
    x = Tensor(generate_ones_input((5, 5), np.float32))
    # input_min & input_max
    bernoulli_forward_func(x, p)
    assert np.allclose(x.asnumpy(), expect, rtol=1e-4)
    grads_x = bernoulli_backward_func(x, p)
    assert np.allclose(grads_x[0].asnumpy(), expect_grad, rtol=1e-4)
    if p_mode == "tensor":
        assert np.allclose(grads_x[1].asnumpy(), expect_grad, rtol=1e-4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("mode", ["pynative", "kbk"])
@pytest.mark.parametrize("p_mode", ["tensor", "float"])
def test_bernoulli_rng_status(mode, p_mode):
    """
    Feature: pyboost function.
    Description: test function Tensor.bernoulli_ with random status.
    Expectation: expect correct result.
    """
    set_context_mode(mode)

    if p_mode == "tensor":
        p = Tensor(generate_ones_input((5, 5), np.float32)) * 0.3
    else:
        p = 0.3

    state = ms.get_rng_state()
    x = Tensor(generate_ones_input((5, 5), np.float32))
    bernoulli_forward_func(x, p)

    y = Tensor(generate_ones_input((5, 5), np.float32))
    bernoulli_forward_func(y, p)

    ms.set_rng_state(state)
    z = Tensor(generate_ones_input((5, 5), np.float32))
    bernoulli_forward_func(z, p)

    assert not (x.asnumpy() == y.asnumpy()).all()
    assert (x.asnumpy() == z.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_bernoulli_dynamic():
    """
    Feature: pyboost function.
    Description: test function Tensor.bernoulli_ with dynamic.
    Expectation: expect correct result.
    """

    state = ms.get_rng_state()
    @test_utils.run_with_cell
    def bernoulli_func(x, p):
        ms.set_rng_state(state)
        y = x * 1
        return y.bernoulli_(p)

    input_seq1 = [Tensor(generate_ones_input((5, 5), np.float32)), 0.3]
    input_seq2 = [Tensor(generate_ones_input((5, 5, 6), np.float32)), 0.7]
    TEST_OP(bernoulli_func, [input_seq1, input_seq2], '', disable_yaml_check=True)

    input_seq3 = [Tensor(generate_ones_input((5, 7, 8), np.float32)),
                  Tensor(generate_ones_input((5, 7, 8), np.float32)) * 0.3]
    input_seq4 = [Tensor(generate_ones_input((6, 4), np.float32)),
                  Tensor(generate_ones_input((6, 4), np.float32)) * 0.7]
    TEST_OP(bernoulli_func, [input_seq3, input_seq4], '', disable_yaml_check=True)
