import numpy as np

import mindspore.context as context
from mindspore import Tensor
from mindspore import jit


@jit
def func(input_x, input_y):
    output = input_x + input_x * input_y
    return output


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor(np.array([1]).astype(np.float32))
    y = Tensor(np.array([2]).astype(np.float32))
    res = func(x, y)
    print("AAA", res, "BBB")
    print("AAA", res.asnumpy().shape, "BBB")
