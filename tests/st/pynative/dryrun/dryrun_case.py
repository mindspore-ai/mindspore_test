import numpy as np
from mindspore import Tensor, ops, context
import mindspore as ms
from mindspore.utils import dryrun
import argparse
import os

context.set_context(mode=ms.PYNATIVE_MODE)
def ops_use_dryrunmock(simulation_flag):
    a = Tensor(np.random.rand(3, 3).astype(np.float32))
    b = ops.matmul(a, a)
    c = Tensor(np.random.randint(1, 100))
    test_ret = []
    test_ret.append(dryrun.mock(np.zeros((3, 3), dtype=np.float32), a.asnumpy))
    test_ret.append(dryrun.mock(np.zeros((3, 3), dtype=np.float32), b.asnumpy))
    test_ret.append(dryrun.mock(0.5, lambda: a[0][0]))
    test_ret.append(dryrun.mock(0.5, lambda: b[0, 0]))
    test_ret.append(dryrun.mock(True, b.is_contiguous))
    test_ret.append(dryrun.mock(50, int, c))
    test_ret.append(dryrun.mock(0.0, float, c))
    test_ret.append(dryrun.mock(np.zeros((3, 3), dtype=np.float32), a.tolist))
    test_ret.append(dryrun.mock(np.zeros((3, 3), dtype=np.float32), a.flush_from_cache))

    target_ret = []
    if simulation_flag:
        target_ret = [np.zeros((3, 3), dtype=np.float32), np.zeros((3, 3), dtype=np.float32), 0.5, 0.5, True, 50, 0.0,
                      np.zeros((3, 3), dtype=np.float32), np.zeros((3, 3), dtype=np.float32)]
    else:
        target_ret = [a.asnumpy(), b.asnumpy(), a[0][0], b[0, 0], b.is_contiguous(), int(c), float(c), a.tolist(),
                      a.flush_from_cache()]

    for idx in range(len(target_ret)):
        assert np.equal(np.array(test_ret[idx]), np.array(target_ret[idx])).all()

def ops_not_use_dryrunmock():
    a = Tensor(np.random.rand(3, 3).astype(np.float32))
    b = ops.matmul(a, a)
    c = Tensor(np.random.randint(1, 100))
    test_ret = []
    test_ret.append(a.asnumpy())
    test_ret.append(b.asnumpy())
    if a[0][0] > 0:
        test_ret.append(a[0][0])
    if b[0, 0] > 0:
        test_ret.append(b[0, 0])
    test_ret.append(b.is_contiguous())
    test_ret.append(int(c) > 50)
    test_ret.append(float(c) > 50)
    test_ret.append(a.tolist())
    test_ret.append(a.flush_from_cache() is not None)
    print(test_ret)

def ops_dryrun_cases(test_case):
    simulation_env = "MS_SIMULATION_LEVEL"
    if test_case == 0:
        if os.environ.get(simulation_env):
            os.environ.pop(simulation_env)
        ops_not_use_dryrunmock()
    elif test_case == 1:
        dryrun.set_simulation()
        ops_not_use_dryrunmock()
    elif test_case == 2:
        dryrun.set_simulation()
        ops_use_dryrunmock(simulation_flag=True)
    elif test_case == 3:
        if os.environ.get(simulation_env):
            os.environ.pop(simulation_env)
        ops_use_dryrunmock(simulation_flag=False)


parser = argparse.ArgumentParser()
parser.add_argument('--test_case', type=int, default=0)
args = parser.parse_args()
ops_dryrun_cases(args.test_case)
