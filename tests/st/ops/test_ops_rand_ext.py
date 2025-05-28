import numpy as np
import pytest

import mindspore as ms
from mindspore.ops.function.random_func import rand_ext, rand_like_ext
from mindspore.ops.function.random_func import randn_ext, randn_like_ext
from mindspore.ops.function.random_func import randint_ext, randint_like_ext

from tests.mark_utils import arg_mark
from tests.st.utils import test_utils


@test_utils.run_with_mode
@test_utils.run_with_cell
def run_rand(*size, dtype=None, generator=None):
    return rand_ext(*size, dtype=dtype, generator=generator)


@test_utils.run_with_mode
@test_utils.run_with_cell
def run_randlike(tensor, dtype=None):
    return rand_like_ext(tensor, dtype=dtype)


@test_utils.run_with_mode
@test_utils.run_with_cell
def run_randn(*size, dtype=None, generator=None):
    return randn_ext(*size, dtype=dtype, generator=generator)


@test_utils.run_with_mode
@test_utils.run_with_cell
def run_randnlike(tensor, dtype=None):
    return randn_like_ext(tensor, dtype=dtype)


@test_utils.run_with_mode
@test_utils.run_with_cell
def run_randint(low, high, size, dtype=None, generator=None):
    return randint_ext(low, high, size, dtype=dtype, generator=generator)


@test_utils.run_with_mode
@test_utils.run_with_cell
def run_randint_default_low_overload(high, size, dtype=None, generator=None):
    return randint_ext(high, size, dtype=dtype, generator=generator)


@test_utils.run_with_mode
@test_utils.run_with_cell
def run_randintlike(tensor, low, high, dtype=None):
    return randint_like_ext(tensor, low, high, dtype=dtype)


@test_utils.run_with_mode
@test_utils.run_with_cell
def run_randint_like_default_low_overload(tensor, high, dtype=None):
    return randint_like_ext(tensor, high, dtype=dtype)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_rand_call(mode):
    """
    Feature: rand, rand_like, randn, randn_like, randint, randint_like function.
    Description: test function call
    Expectation: expect correct result.
    """
    # rand, randlike
    shape = (5, 5)
    x = run_rand(*shape, dtype=ms.float64, mode=mode).asnumpy()
    x2 = run_rand(shape, dtype=ms.float64, mode=mode).asnumpy()
    y = run_randlike(ms.Tensor(np.random.randn(*shape)),
                     dtype=ms.float64, mode=mode).asnumpy()
    assert np.all((x < 1) & (x >= 0))
    assert np.all((x2 < 1) & (x2 >= 0))
    assert np.all((y < 1) & (y >= 0))
    assert x.dtype == np.float64
    assert x2.dtype == np.float64
    assert y.dtype == np.float64
    assert x.shape == shape
    assert x2.shape == shape
    assert y.shape == shape

    # randn, randn_like
    shape = (7, 8, 9)
    x = run_randn(*shape, dtype=ms.float64, mode=mode).asnumpy()
    x2 = run_randn(shape, dtype=ms.float64, mode=mode).asnumpy()
    y = run_randnlike(ms.Tensor(np.random.randn(*shape)),
                      dtype=ms.float64, mode=mode).asnumpy()
    assert x.dtype == np.float64
    assert x2.dtype == np.float64
    assert y.dtype == np.float64
    assert x.shape == shape
    assert x2.shape == shape
    assert y.shape == shape

    # randint, randint_like
    low = -10
    high = 10
    shape = (2, 3, 4, 5)
    dtype = ms.int32
    x = run_randint(low, high, shape, dtype=dtype, mode=mode).asnumpy()
    y = run_randintlike(ms.Tensor(np.random.randn(*shape)), low, high,
                        dtype=dtype, mode=mode).asnumpy()
    assert np.all((x >= low) & (x < high))
    assert np.all((y >= low) & (y < high))
    assert x.dtype == np.int32
    assert y.dtype == np.int32
    assert x.shape == shape
    assert y.shape == shape

    # test randint/randint_like overload with default low=0
    x = run_randint_default_low_overload(high, shape, dtype=dtype, mode=mode).asnumpy()
    assert np.all(x >= 0)
    y = run_randint_like_default_low_overload(
        ms.Tensor(x), 100, dtype=dtype, mode=mode).asnumpy()
    assert np.all(y >= 0)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_rand_randomness(mode):
    """
    Feature: rand function.
    Description: test randomness of rand
    Expectation: expect correct result.
    """
    generator = ms.Generator()
    generator.seed()

    shape = (5, 5)
    x1 = run_rand(*shape, generator=generator, mode=mode).asnumpy()
    x2 = run_rand(*shape, generator=generator, mode=mode).asnumpy()
    y1 = run_randn(*shape, generator=generator, mode=mode).asnumpy()
    y2 = run_randn(*shape, generator=generator, mode=mode).asnumpy()
    z1 = run_randint(0, 10, shape, generator=generator, mode=mode).asnumpy()
    z2 = run_randint(0, 10, shape, generator=generator, mode=mode).asnumpy()

    assert np.any(x1 != x2)
    assert np.any(y1 != y2)
    assert np.any(z1 != z2)

    state = generator.get_state()
    x1 = run_rand(*shape, generator=generator, mode=mode).asnumpy()
    y1 = run_randn(*shape, generator=generator, mode=mode).asnumpy()
    z1 = run_randint(0, 10, shape, generator=generator, mode=mode).asnumpy()
    generator.set_state(state)
    x2 = run_rand(*shape, generator=generator, mode=mode).asnumpy()
    y2 = run_randn(*shape, generator=generator, mode=mode).asnumpy()
    z2 = run_randint(0, 10, shape, generator=generator, mode=mode).asnumpy()

    assert np.all(x1 == x2)
    assert np.all(y1 == y2)
    assert np.all(z1 == z2)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_randlike_randomness(mode):
    """
    Feature: randlike function.
    Description: test randomness of rand_like
    Expectation: expect correct result.
    """
    tensor = ms.Tensor(np.random.randn(5, 5))
    x1 = run_randlike(tensor, mode=mode).asnumpy()
    x2 = run_randlike(tensor, mode=mode).asnumpy()
    y1 = run_randnlike(tensor, mode=mode).asnumpy()
    y2 = run_randnlike(tensor, mode=mode).asnumpy()
    z1 = run_randintlike(tensor, 0, 10, mode=mode).asnumpy()
    z2 = run_randintlike(tensor, 0, 10, mode=mode).asnumpy()

    assert np.any(x1 != x2)
    assert np.any(y1 != y2)
    assert np.any(z1 != z2)

    state = ms.get_rng_state()
    x1 = run_randlike(tensor, mode=mode).asnumpy()
    y1 = run_randnlike(tensor, mode=mode).asnumpy()
    z1 = run_randintlike(tensor, 0, 10, mode=mode).asnumpy()
    ms.set_rng_state(state)
    x2 = run_randlike(tensor, mode=mode).asnumpy()
    y2 = run_randnlike(tensor, mode=mode).asnumpy()
    z2 = run_randintlike(tensor, 0, 10, mode=mode).asnumpy()

    assert np.all(x1 == x2)
    assert np.all(y1 == y2)
    assert np.all(z1 == z2)
