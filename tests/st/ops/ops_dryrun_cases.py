# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import mindspore as ms
import numpy as np
import os


def ops_dryrun_check_results(dryrun_check, no_dryrun_check, *args, **kwargs):
    if os.environ.get("MS_SIMULATION_LEVEL"):
        dryrun_check(*args, **kwargs)
    else:
        no_dryrun_check(*args, **kwargs)


def ops_dryrun_check_ndarray_result(result, expect, rtol=1e-5, atol=1e-8, equal_nan=False, *, shape_cmp=True,
                                    dtype_cmp=True):

    def dryrun_check_result(result, expect, shape_cmp, dtype_cmp):
        if isinstance(result, (tuple, list)) and isinstance(expect, (tuple, list)):
            for result_i, expect_i in zip(result, expect):
                dryrun_check_result(result_i, expect_i, shape_cmp, dtype_cmp)
        else:
            if isinstance(result, ms.Tensor):
                result = result.asnumpy()
            if isinstance(expect, ms.Tensor):
                expect = expect.asnumpy()
            if shape_cmp:
                np.testing.assert_equal(result.shape, expect.shape)
            if dtype_cmp:
                np.testing.assert_equal(result.dtype, expect.dtype)

    def nodryrun_check_result(result, expect, rtol, atol, equal_nan):
        if isinstance(result, (tuple, list)) and isinstance(expect, (tuple, list)):
            for result_i, expect_i in zip(result, expect):
                nodryrun_check_result(result_i, expect_i, rtol, atol, equal_nan)
        else:
            if isinstance(result, ms.Tensor):
                result = result.asnumpy()
            if isinstance(expect, ms.Tensor):
                expect = expect.asnumpy()
            np.testing.assert_allclose(result, expect, rtol=rtol, atol=atol, equal_nan=equal_nan)

    if os.environ.get("MS_SIMULATION_LEVEL"):
        dryrun_check_result(result, expect, shape_cmp, dtype_cmp)
    else:
        nodryrun_check_result(result, expect, rtol, atol, equal_nan)
