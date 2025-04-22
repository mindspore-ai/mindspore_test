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
    @ms.jit(dynamic=1)
    @ms.enable_dynamic(y=ms.Tensor(shape=None, dtype=ms.float32),
                       x=ms.Tensor(shape=[2, None], dtype=ms.float32))
    def fn(x, y, z):
        return x + 1, y + 1, z + 1

    x1 = ms.Tensor(np.random.randn(2, 1), ms.float32)
    x2 = ms.Tensor(np.random.randn(2, 2), ms.float32)
    x3 = ms.Tensor(np.random.randn(2, 3), ms.float32)
    x4 = ms.Tensor(np.random.randn(2, 4), ms.float32)

    y1 = ms.Tensor(np.random.randn(2, 3), ms.float32)
    y2 = ms.Tensor(np.random.randn(3, 4, 5), ms.float32)
    y3 = ms.Tensor(np.random.randn(4, 5, 6, 7), ms.float32)
    y4 = ms.Tensor(np.random.randn(5, 6, 7, 8, 9), ms.float32)

    z1 = ms.Tensor(np.random.randn(2, 1), ms.float32)
    z2 = ms.Tensor(np.random.randn(2, 2), ms.float32)
    z3 = ms.Tensor(np.random.randn(2, 3), ms.float32)
    z4 = ms.Tensor(np.random.randn(2, 4), ms.float32)

    fn(x1, y1, z1)
    fn(x2, y2, z2)
    fn(x3, y3, z3)
    fn(x4, y4, z4)


def run_fn2():
    @ms.jit(dynamic=1)
    @ms.enable_dynamic(x=[ms.Tensor(shape=[2, None], dtype=ms.float32), ms.Tensor(shape=[2, 2], dtype=ms.float32)])
    def fn(x, y):
        return x[0] + 1, y + 1

    x1 = ms.Tensor(np.random.randn(2, 1), ms.float32)
    x2 = ms.Tensor(np.random.randn(2, 2), ms.float32)
    x3 = ms.Tensor(np.random.randn(2, 3), ms.float32)
    x4 = ms.Tensor(np.random.randn(2, 4), ms.float32)

    y1 = ms.Tensor(np.random.randn(2, 1), ms.float32)
    y2 = ms.Tensor(np.random.randn(2, 2), ms.float32)
    y3 = ms.Tensor(np.random.randn(2, 3), ms.float32)
    y4 = ms.Tensor(np.random.randn(2, 4), ms.float32)

    list1 = [x1, x2]
    list2 = [x2, x2]
    list3 = [x3, x2]
    list4 = [x4, x2]

    fn(ms.mutable(list1), y1)
    fn(ms.mutable(list2), y2)
    fn(ms.mutable(list3), y3)
    fn(ms.mutable(list4), y4)



if __name__ == "__main__":
    fn_name = sys.argv[1]
    if fn_name == "fn1":
        run_fn1()
    elif fn_name == "fn2":
        run_fn2()
