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
"""SDC detect."""
from mindspore import _c_expression


def sdc_detect_start():
    """
    Start silent data corruption detection. It will check the inputs and outputs of MatMul operations during the
    forward and backward computations on the current device, which may increase execution time. The overhead of the
    check time decreases as the matrix shapes increase. Starting sdc detection results in approximately 100%
    performance degradation for a single 4096-sized MatMul computation, and approximately 90% degradation on the
    Llama2-7B model (model parallel is 4, pipeline parallel is 2, and using qkv concatenation and ffn concatenation in
    decoder layers).

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindspore.utils import sdc_detect_start
        >>> sdc_detect_start()
    """
    _c_expression.sdc_detect_start()


def sdc_detect_stop():
    """
    Stop silent data corruption detection.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindspore.utils import sdc_detect_stop
        >>> sdc_detect_stop()
    """
    _c_expression.sdc_detect_stop()


def get_sdc_detect_result():
    """
    Get the result of silent data corruption detection.

    Returns:
        bool, indicating whether silent data corruption has occurred after detection start.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindspore.utils import get_sdc_detect_result
        >>> result = get_sdc_detect_result()
        >>> print(result)
        False
    """
    return _c_expression.get_sdc_detect_result()
