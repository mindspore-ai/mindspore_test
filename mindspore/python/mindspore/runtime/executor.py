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
from mindspore._c_expression import RuntimeConf
from mindspore.device_manager import _check_runtime_conf_env_valid
from mindspore._checkparam import args_type_check
from mindspore import _checkparam as Validator

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
