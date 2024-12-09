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

"""Executor manager interfaces."""
from mindspore._c_expression import RuntimeConf, RuntimeExecutor
from mindspore.device_manager import _check_runtime_conf_env_valid
from mindspore.runtime.thread_bind_core import _get_cpu_affinity_policy
from mindspore._checkparam import args_type_check
from mindspore import _checkparam as Validator
from mindspore import log as logger


def launch_blocking():
    """
    Control the execution mode of device operations.

    Note:
        - No parameters are required.
        - By default, operations are executed asynchronously.
        - Calling this function enables synchronous execution.

    Examples:
        >>> import mindspore as ms
        >>> ms.set_device("Ascend", 1)
        >>> ms.runtime.launch_blocking()
    """
    return RuntimeConf.get_instance().set_launch_blocking()

@args_type_check(threads_num=int)
def dispatch_threads_num(threads_num):
    """
    Set the threads number of runtime used.

    Args:
        threads_num (int): The threads number of runtime used. Default: ``5``.

    Examples:
        >>> import mindspore as ms
        >>> ms.set_device("Ascend", 1)
        >>> ms.runtime.dispatch_threads_num(6)
    """
    _check_runtime_conf_env_valid()
    if RuntimeConf.get_instance().is_dispatch_threads_num_configured():
        raise RuntimeError("The 'dispatch_threads_num' can not be set repeatedly.")

    threads_num = Validator.check_positive_int(threads_num, "threads_num")

    return RuntimeConf.get_instance().set_dispatch_threads_num(threads_num)


@args_type_check(enable_affinity=bool, affinity_cpu_list=dict)
def set_cpu_affinity(enable_affinity, affinity_cpu_list=None):
    """
    Enable binding cpu core to specific thread according to devices' NUMA affinity.

    Note:
        - If `affinity_cpu_list` is not specified, the cpu range binding to each device will be assigned automatically.

    Args:
        enable_affinity (bool): Determine whether enable cpu affinity binding, should be 'true' or 'false'.
        affinity_cpu_list (str, optional): Specify cpu ranges for the device.

    Returns:
        None

    Examples:
        >>> import mindspore as ms
        >>> ms.set_device("Ascend", 1)
        >>> ms.runtime.set_cpu_affinity(true)

        >>> import mindspore as ms
        >>> ms.set_device("Ascend", 1)
        >>> ms.runtime.set_cpu_affinity(true, {"device0":["0-9"],"device1":["10-15","20-29"],"device2":["35-40"]})
    """
    if RuntimeExecutor.get_instance().is_thread_bind_core_configured():
        raise RuntimeError("The 'mindspore.runtime.set_cpu_affinity' cannot be set repeatedly.")
    if enable_affinity:
        module_bind_core_policy, bind_policy_flag = _get_cpu_affinity_policy(affinity_cpu_list)
        if not module_bind_core_policy:
            logger.warning("set_cpu_affinity is not enabled because the environment does not meet the "
                           "basic conditions for binding core.")
            RuntimeExecutor.get_instance().set_thread_bind_core_configured()
            return
        if bind_policy_flag:
            RuntimeExecutor.get_instance().thread_bind_core_with_policy(module_bind_core_policy)
        else:
            RuntimeExecutor.get_instance().thread_bind_core(module_bind_core_policy)
    else:
        RuntimeExecutor.get_instance().set_thread_bind_core_configured()
        return
