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

"""Op debug interfaces."""
from mindspore._checkparam import args_type_check
from  .device import _is_supported
try:
    from mindspore._c_expression import AscendOpDebugConf
except ImportError:
    pass

function_status = {'execute_timeout': False, 'debug_option': False}

@args_type_check(op_timeout=int)
def execute_timeout(op_timeout):
    """
    Set the maximum duration of executing an operator in seconds. The framework operator execution timeout time
    is ``900`` by default.
    please refer to `Ascend Community document about aclrtSetOpExecuteTimeOut
    <https://www.hiascend.com/document/detail/en/CANNCommunityEdition/600alphaX/infacldevg/aclcppdevg/aclcppdevg_03_0069.html>`_.

    Args:
        op_timeout (int): Set the maximum duration of executing an operator in seconds.
          If the execution time exceeds this value, system will terminate the task.
          0 means endless wait. The defaults for AI Core and AICPU operators vary on different hardware.

    Examples:
        >>> import mindspore as ms
        >>> ms.device_context.ascend.op_debug.execute_timeout(100)
    """
    if not function_status['execute_timeout']:
        function_status['execute_timeout'] = True
        if not _is_supported():
            return
    if op_timeout == AscendOpDebugConf.get_instance().execute_timeout():
        return
    # Check the configuration environment whether valid
    if AscendOpDebugConf.get_instance().is_execute_timeout_configured():
        raise RuntimeError("The 'execute_timeout' can not be set repeatedly.")
    if op_timeout < 0:
        raise ValueError("The num of execute_timeout must bigger than or equal to 0.")
    AscendOpDebugConf.get_instance().set_execute_timeout(op_timeout)


def debug_option(option_value):
    """
    Enable debugging options for Ascend operators, default not enabled.

    Args:
        option_value(str): Ascend operators debugging configuration. Currently, only memory
            access violation detection is supported.
            The value currently only supports being set to ``"oom"``.

            - ``"oom"``: When there is a memory out of bounds during the execution of an operator,
              AscendCL will return an error code of ``EZ9999``.

    Examples:
        >>> import mindspore as ms
        >>> ms.device_context.ascend.op_debug.debug_option("oom")
    """
    if not function_status['debug_option']:
        function_status['debug_option'] = True
        if not _is_supported():
            return
    if option_value == AscendOpDebugConf.get_instance().debug_option():
        return
    # Check the configuration environment whether valid
    if AscendOpDebugConf.get_instance().is_debug_option_configured():
        raise RuntimeError("The 'debug_option' can not be set repeatedly.")
    valid_order = {"oom"}
    if not isinstance(option_value, str):
        raise TypeError(
            f"For 'device_context.ascend.op_debug.debug_option(option_value)', the type of 'option_value' must be str, "
            f"but got {type(option_value)}."
        )
    if option_value not in valid_order:
        raise ValueError(
            f"For 'device_context.ascend.op_debug.debug_option(option_value)', the 'option_value' supports being set "
            f"to 'oom' currently, but got {option_value}."
        )
    AscendOpDebugConf.get_instance().set_debug_option(option_value)
