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

"""Memory interfaces."""

from mindspore._c_expression import RuntimeConf
from mindspore import _checkparam as Validator
from mindspore.device_manager import _check_runtime_conf_env_valid
from mindspore._checkparam import args_type_check

_MEMORY_PATTERN = r'[1-9][0-9]*(\.)?[0-9]*GB|0\.[0-9]*GB'

@args_type_check(init_size=str, increase_size=str, max_size=str, optimize_level=str)
def set_memory(init_size, increase_size, max_size, optimize_level):
    """
    Set the memory parameters of runtime device memory management that is implemented using a memory pool.

    Args:
        init_size (string): The init size of memory pool. The format is "xxGB", Default: ``2G`` .
        increase_size (string): The increase size of memory pool.When the current memory pool has no
            enough memory, the memory pool will be expanded by this value. The format is "xxGB", Default: ``2G`` .
        max_size (string): The maximum memory available for memory pool.
            The actual used memory size is the minimum of the available memory of the device and max_device_memory.
            The format is "xxGB", Default is the maximum available memory of the device, expressed as ``1024G``.
        optimize_level (string): The memory optimize level. The value must be in ['O0', 'O1'], Default: ``O0`` .

    Examples:
        >>> import mindspore as ms
        >>> ms.set_device("Ascend", 1)
        >>> ms.runtime.set_memory("10G", "2G", "60G", "O1")
    """
    _check_runtime_conf_env_valid()
    if RuntimeConf.get_instance().is_memory_configured():
        raise RuntimeError("The 'set_memory' can not be set repeatedly.")

    _check_memory_conf_valid(init_size)
    _check_memory_conf_valid(increase_size)
    _check_memory_conf_valid(max_size)
    init_value = float(init_size[:-2])
    increase_value = float(increase_size[:-2])
    max_value = float(max_size[:-2])

    memory_optimize_levels = ["O0", "O1"]
    if optimize_level not in memory_optimize_levels:
        raise ValueError(f"The optimize_level must be one of "
                         f"{memory_optimize_levels}, but got {optimize_level}.")
    optimize_value = 0
    if optimize_level == "O1":
        optimize_value = 1

    return RuntimeConf.get_instance().set_memory(init_value, increase_value, max_value, optimize_value)

def _check_memory_conf_valid(memory_size):
    """
    Check whether the configuration memory value format is "xxGB" and can not be "0G".
    """
    if not Validator.check_str_by_regular(memory_size, _MEMORY_PATTERN):
        raise ValueError("The memory value should be in correct format!"
                         "It must be a string ending with 'GB', in addition to that, it must contain "
                         "only numbers or decimal points, such as \"5GB\" or \"3.5GB\", but got {}."
                         .format(memory_size))
    if memory_size == "0G" or memory_size == "0.0G":
        raise ValueError("The memory value should not be \"0GB\".")
