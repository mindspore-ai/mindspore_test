import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor, context
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.mark_utils import arg_mark


class Net(ms.nn.Cell):
    def construct(self, input_x):
        input_x.erfinv_()
        return input_x

def erfinv__forward_func(input_x):
    return Net()(input_x)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative'])
def test_t_erfinv__normal(mode):
    """
    Feature: erfinv_ function.
    Description: test function erfinv_
    Expectation: expect correct result.
    """
    # Tensor.erfinv_ forward
    shape = (5, 5)
    input_x = ms.Tensor(np.ones(shape), dtype=ms.float32) * 0.5
    expect = np.array([[0.47693613, 0.47693613, 0.47693613, 0.47693613, 0.47693613],
                       [0.47693613, 0.47693613, 0.47693613, 0.47693613, 0.47693613],
                       [0.47693613, 0.47693613, 0.47693613, 0.47693613, 0.47693613],
                       [0.47693613, 0.47693613, 0.47693613, 0.47693613, 0.47693613],
                       [0.47693613, 0.47693613, 0.47693613, 0.47693613, 0.47693613]])
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
        output = erfinv__forward_func(input_x).asnumpy()
    else:
        context.set_context(mode=ms.GRAPH_MODE)
        output = erfinv__forward_func(input_x).asnumpy()
    assert output.dtype == np.float32
    assert np.allclose(output, expect)
    assert np.all(output == input_x.asnumpy())

    # in-place ops does not support backward now

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_t_erfinv__dynamic():
    """
    Feature: test dynamic erfinv_.
    Description: test dynamic of op InplaceErfinv.
    Expectation: expect correct result.
    """
    input_1 = Tensor(np.zeros((5, 5)), dtype=ms.float32)
    input_2 = Tensor(np.ones((3, 4, 5)), dtype=ms.float32)
    # dynamic string is not supported
    TEST_OP(erfinv__forward_func, [[input_1], [input_2]], 'inplace_erfinv', disable_grad=True,
            disable_yaml_check=True, disable_mode=["GRAPH_MODE"])
