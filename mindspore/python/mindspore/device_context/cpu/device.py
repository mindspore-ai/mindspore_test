# Copyright 2024 Huawei Technologies Co., Ltd
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

"""The CPU device interfaces."""
from mindspore._c_expression import MSContext
from mindspore import log as logger

def is_available():
    """
    Returns whether the CPU of this MindSpore package is available.
    All dependent libraries should be successfully loaded if this CPU is available.

    Inputs:
        No inputs.

    Returns:
        Bool, whether the CPU is available for this MindSpore package.

    Examples:
        >>> import mindspore as ms
        >>> print(ms.device_context.cpu.is_available())
        True
    """
    # MindSpore will try to load plugins in "import mindspore", and availability status will be stored.
    context = MSContext.get_instance()
    if not context.is_pkg_support_device("CPU"):
        logger.warning(f"The CPU device is not available.")
        load_plugin_error = context.load_plugin_error()
        if load_plugin_error != "":
            logger.warning(
                f"Here's error when loading plugin for MindSpore package."
                f"Error message: {load_plugin_error}"
            )
        return False
    return True

def device_count():
    """
    Get the CPU device count. Always return 1 for CPU.

    Returns:
        int.

    Inputs:
        No inputs.

    Examples:
        >>> import mindspore as ms
        >>> print(ms.device_context.cpu.device_count())
        1
    """
    return 1
