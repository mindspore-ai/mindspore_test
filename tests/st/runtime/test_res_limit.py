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

import mindspore.context as context
from mindspore import Tensor, ops
import mindspore as ms
from tests.mark_utils import arg_mark
ms.set_device("Ascend")
context.set_context(mode=context.PYNATIVE_MODE)

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
def test_runtime_get_device_limit():
    """
    Feature: runtime stream api.
    Description: Test runtime.get_device_limit api.
    Expectation: runtime.get_device_limit api performs as expected.
    """
    ret = ms.runtime.get_device_limit(0)
    print(ret)
    assert ret["cube_core_num"] == 24
    assert ret["vector_core_num"] == 48
    ms.runtime.synchronize()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
def test_runtime_set_device_limit():
    """
    Feature: runtime stream api.
    Description: Test runtime.set_device_limit api.
    Expectation: runtime.set_device_limit api performs as expected.
    """
    ret = ms.runtime.get_device_limit(0)
    assert ret["cube_core_num"] == 24
    assert ret["vector_core_num"] == 48

    ms.runtime.set_device_limit(0, 8, 8)
    ret = ms.runtime.get_device_limit(0)
    assert ret["cube_core_num"] == 8
    assert ret["vector_core_num"] == 8

    ms.runtime.set_device_limit(0, 4, 4)
    ret = ms.runtime.get_device_limit(0)
    assert ret["cube_core_num"] == 4
    assert ret["vector_core_num"] == 4

    ms.runtime.set_device_limit(0, 3, -1)
    ret = ms.runtime.get_device_limit(0)
    assert ret["cube_core_num"] == 3
    assert ret["vector_core_num"] == 4

    ms.runtime.set_device_limit(0, -1, 3)
    ret = ms.runtime.get_device_limit(0)
    assert ret["cube_core_num"] == 3
    assert ret["vector_core_num"] == 3

    ms.runtime.set_device_limit(0, -1, -1)
    ret = ms.runtime.get_device_limit(0)
    assert ret["cube_core_num"] == 3
    assert ret["vector_core_num"] == 3
    ms.runtime.synchronize()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
def test_runtime_get_stream_limit():
    """
    Feature: runtime stream api.
    Description: Test runtime.get_stream_limit api.
    Expectation: runtime.get_stream_limit api performs as expected.
    """
    s1 = ms.runtime.Stream()
    ret = ms.runtime.get_stream_limit(s1)
    assert ret["cube_core_num"] == 24
    assert ret["vector_core_num"] == 48
    ms.runtime.synchronize()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
def test_runtime_set_stream_limit():
    """
    Feature: runtime stream api.
    Description: Test runtime.set_stream_limit api.
    Expectation: runtime.set_stream_limit api performs as expected.
    """
    s1 = ms.runtime.Stream()
    ret = ms.runtime.get_stream_limit(s1)
    assert ret["cube_core_num"] == 24
    assert ret["vector_core_num"] == 48

    ms.runtime.set_stream_limit(s1, 8, 8)
    ret = ms.runtime.get_stream_limit(s1)
    assert ret["cube_core_num"] == 8
    assert ret["vector_core_num"] == 8

    ms.runtime.set_stream_limit(s1, 4, 4)
    ret = ms.runtime.get_stream_limit(s1)
    assert ret["cube_core_num"] == 4
    assert ret["vector_core_num"] == 4

    ms.runtime.set_stream_limit(s1, 3, -1)
    ret = ms.runtime.get_stream_limit(s1)
    assert ret["cube_core_num"] == 3
    assert ret["vector_core_num"] == 4

    ms.runtime.set_stream_limit(s1, -1, 3)
    ret = ms.runtime.get_stream_limit(s1)
    assert ret["cube_core_num"] == 3
    assert ret["vector_core_num"] == 3

    ms.runtime.set_stream_limit(s1, -1, -1)
    ret = ms.runtime.get_stream_limit(s1)
    assert ret["cube_core_num"] == 3
    assert ret["vector_core_num"] == 3
    ms.runtime.synchronize()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
def test_runtime_reset_stream_limit():
    """
    Feature: runtime stream api.
    Description: Test runtime.reset_stream_limit api.
    Expectation: runtime.reset_stream_limit api performs as expected.
    """
    s1 = ms.runtime.Stream()
    ret = ms.runtime.get_stream_limit(s1)
    assert ret["cube_core_num"] == 24
    assert ret["vector_core_num"] == 48

    ms.runtime.set_stream_limit(s1, 8, 8)
    ret = ms.runtime.get_stream_limit(s1)
    assert ret["cube_core_num"] == 8
    assert ret["vector_core_num"] == 8

    ms.runtime.reset_stream_limit(s1)
    ret = ms.runtime.get_stream_limit(s1)
    assert ret["cube_core_num"] == 24
    assert ret["vector_core_num"] == 48
    ms.runtime.synchronize()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
def test_runtime_stream_limit_ctx():
    """
    Feature: runtime stream api.
    Description: Test runtime.StreamLimitCtx api.
    Expectation: runtime.StreamLimitCtx api performs as expected.
    """
    a = Tensor(2.0)
    s1 = ms.runtime.Stream()
    ret = ms.runtime.get_stream_limit(s1)
    assert ret["cube_core_num"] == 24
    assert ret["vector_core_num"] == 48
    with ms.runtime.StreamCtx(s1):
        with ms.runtime.StreamLimitCtx(s1, 8, 8):
            ret = ms.runtime.get_stream_limit(s1)
            assert ret["cube_core_num"] == 8
            assert ret["vector_core_num"] == 8
            ops.abs(a)
            ret = ms.runtime.get_stream_limit(s1)
            assert ret["cube_core_num"] == 8
            assert ret["vector_core_num"] == 8
    ret = ms.runtime.get_stream_limit(s1)
    assert ret["cube_core_num"] == 24
    assert ret["vector_core_num"] == 48
    s1.synchronize()
