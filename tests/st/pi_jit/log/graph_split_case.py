"""pijit graph split helper script (migrated)"""

import pytest
import numpy as np

from mindspore import jit, ops
from mindspore.common import Tensor


def test_pijit():
    """
    Feature: Test pijit log (bytecode only).
    Description: Enable bytecode log and run a graph-split helper script.
    Expectation: Expect to see bytecode log content.
    Migrated from: test_parse_pijit_improve_debug_ability.py::test_parse_pijit_improve_debug_ability_001
    """
    def fn(x):
        # graph break by converting Tensor to numpy
        x_np = x.asnumpy()
        y = x_np * x_np
        z = Tensor(y)
        out = ops.div(z, z)
        return Tensor(out.asnumpy())

    x = ops.randn(4)
    o1 = fn(x)

    compiled_fn = jit(fn, capture_mode='bytecode')
    o2 = compiled_fn(x)

    assert np.allclose(o1.asnumpy(), o2.asnumpy())


if __name__ == "__main__":
    pytest.main(["-vs", __file__])
