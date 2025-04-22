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
import sys
import numpy as np
import mindspore as ms


def run_fn1():
    @ms.jit
    @ms.enable_dynamic(y=ms.Tensor(shape=[None, None], dtype=ms.float32),
                       z=ms.Tensor(shape=[2, None], dtype=ms.float32))
    def fn(x, y, z):
        return x + 1, y + 1, z + 1

    x1 = ms.Tensor(np.random.randn(2, 3), ms.float32)
    x2 = ms.Tensor(np.random.randn(2, 3), ms.float32)
    x3 = ms.Tensor(np.random.randn(2, 3), ms.float32)

    y1 = ms.Tensor(np.random.randn(2, 3), ms.float32)
    y2 = ms.Tensor(np.random.randn(3, 2), ms.float32)
    y3 = ms.Tensor(np.random.randn(4, 4), ms.float32)

    z1 = ms.Tensor(np.random.randn(2, 1), ms.float32)
    z2 = ms.Tensor(np.random.randn(2, 2), ms.float32)
    z3 = ms.Tensor(np.random.randn(2, 3), ms.float32)

    fn(x1, y1, z1)
    fn(x2, y2, z2)
    fn(x3, y3, z3)


def run_fn2():
    @ms.enable_dynamic(x=ms.Tensor(shape=None, dtype=ms.float32),
                       a=ms.Tensor(shape=[2, None], dtype=ms.float32),
                       y=ms.Tensor(shape=None, dtype=ms.float32))
    @ms.jit
    def fn(x, y, a, b, *args, **kwargs):
        return x + 1, y + 1, a + b, args[0] + args[1]

    x1 = ms.Tensor(np.random.randn(1, 1), ms.float32)
    x2 = ms.Tensor(np.random.randn(1, 2), ms.float32)
    x3 = ms.Tensor(np.random.randn(1, 2, 3), ms.float32)

    y1 = ms.Tensor(np.random.randn(2, 2), ms.float32)
    y2 = ms.Tensor(np.random.randn(1), ms.float32)
    y3 = ms.Tensor(np.random.randn(1, 2, 3, 4), ms.float32)

    a1 = ms.Tensor(np.random.randn(2, 1), ms.float32)
    a2 = ms.Tensor(np.random.randn(2, 2), ms.float32)
    a3 = ms.Tensor(np.random.randn(2, 3), ms.float32)

    t1 = ms.Tensor(np.random.randn(3, 3), ms.float32)
    t2 = ms.Tensor(np.random.randn(3, 3), ms.float32)
    t3 = ms.Tensor(np.random.randn(3, 3), ms.float32)

    fn(x1, y1, a1, 1, t1, t1)
    fn(x2, y2, a2, 1, t2, t2)
    fn(x3, y3, a3, 1, t3, t3)


def run_fn3():
    def _fn(a, b, c):
        return a + 1, b + 1, c + 1
    fn = ms.enable_dynamic(a=ms.Tensor(shape=None, dtype=ms.float32),
                           b=ms.Tensor(shape=None, dtype=ms.float32),
                           c=ms.Tensor(shape=None, dtype=ms.float32))(ms.jit(_fn))

    x1 = ms.Tensor(np.random.randn(1, 1), ms.float32)
    x2 = ms.Tensor(np.random.randn(1, 2), ms.float32)
    x3 = ms.Tensor(np.random.randn(1, 2, 3), ms.float32)

    y1 = ms.Tensor(np.random.randn(2, 2), ms.float32)
    y2 = ms.Tensor(np.random.randn(1), ms.float32)
    y3 = ms.Tensor(np.random.randn(1, 2, 3, 4), ms.float32)

    z1 = ms.Tensor(np.random.randn(2), ms.float32)
    z2 = ms.Tensor(np.random.randn(3), ms.float32)
    z3 = ms.Tensor(np.random.randn(2, 3), ms.float32)

    fn(x1, y1, z1)
    fn(x2, y2, z2)
    fn(x3, y3, z3)


def run_fn4():
    @ms.jit
    @ms.enable_dynamic(y=[ms.Tensor(shape=[None, 1], dtype=ms.float32), ms.Tensor(shape=[2, None], dtype=ms.float32)])
    def fn(x, y):
        return x + 1, y[0] + 1, y[1] + 1

    x1 = ms.Tensor(np.random.randn(2, 3), ms.float32)
    x2 = ms.Tensor(np.random.randn(2, 3), ms.float32)
    x3 = ms.Tensor(np.random.randn(2, 3), ms.float32)

    y1 = ms.Tensor(np.random.randn(2, 1), ms.float32)
    y2 = ms.Tensor(np.random.randn(3, 1), ms.float32)
    y3 = ms.Tensor(np.random.randn(4, 1), ms.float32)

    z1 = ms.Tensor(np.random.randn(2, 1), ms.float32)
    z2 = ms.Tensor(np.random.randn(2, 2), ms.float32)
    z3 = ms.Tensor(np.random.randn(2, 3), ms.float32)

    fn(x1, ms.mutable([y1, z1]))
    fn(x2, ms.mutable([y2, z2]))
    fn(x3, ms.mutable([y3, z3]))


if __name__ == "__main__":
    fn_name = sys.argv[1]
    if fn_name == "fn1":
        run_fn1()
    elif fn_name == "fn2":
        run_fn2()
    elif fn_name == "fn3":
        run_fn3()
    elif fn_name == "fn4":
        run_fn4()
