# coding=utf-8

import mindspore as ms
from mindspore import ops, jit
from mindspore.common import Tensor
from mindspore.common import dtype as mstype
import numpy as np

from tests.mark_utils import arg_mark
from ..share.utils import match_array, assert_executed_by_graph_mode


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_shape_not_none():
    '''
    Description:
        1. create a tensor, all args are int
    Expectation:
        1. the net run ok
        2. the result is the same as psjit
    '''
    Tensor(input_data=None, dtype=mstype.float32, shape=[2, 4], init=1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_shape():
    '''
    Description:
        1. create a tensor, all args are None
    Expectation:
        1. the net run ok
        2. the result is the same as psjit
    '''
    x = Tensor(dtype=mstype.float32, shape=[None, 4])
    s = x.shape
    assert s == (-1, 4)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_create_new_tensor_from_dynamic_shape_tensor():
    """
    Feature: Test dynamic shape tensor.
    Description: Use ms.tensor() to create a tensor from a dynamic shape tensor.
    Expectation: No graph break, no exception.
    """

    def fn(x: Tensor):
        y = ops.nonzero(x, as_tuple=False)  # it will create a dynamic shape tensor
        z = ms.tensor(y)
        return z

    x = Tensor(np.array([1, 0, 2, 0, 3]), ms.int32)
    o1 = fn(x)
    # Set print_after_all=True, to check printing dynamic shape tensor will not throw exception.
    fn = jit(fn, mode='PIJit', jit_config={'print_after_all': True, 'compile_with_try': False})
    o2 = fn(x)

    assert_executed_by_graph_mode(fn)
