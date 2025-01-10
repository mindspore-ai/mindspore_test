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

from mindspore._c_expression import RuntimeConf, DeviceManagerConf, _memory_stats, \
    _reset_max_mem_reserved, _reset_max_mem_allocated, DeviceContextManager
from mindspore import _checkparam as Validator
from mindspore._checkparam import args_type_check
from mindspore import log as logger
import mindspore as ms

_MEMORY_PATTERN = r'[1-9][0-9]*(\.)?[0-9]*GB|0\.[0-9]*GB'
_device_context_mgr = DeviceContextManager.get_instance()


@args_type_check(init_size=str, increase_size=str, max_size=str, optimize_level=str)
def set_memory(init_size="2GB", increase_size="2GB", max_size="1024GB", optimize_level="O0"):
    """
    Set the memory parameters of runtime device memory management that is implemented using a memory pool.

    The framework will set all the args by default as follows.

    Args:
        init_size (str): The init size of memory pool. The format is "xxGB". Default: ``2GB`` .
        increase_size (str): The increase size of memory pool. When the current memory pool has no
            enough memory, the memory pool will be expanded by this value. The format is "xxGB". Default: ``2GB`` .
        max_size (str): The maximum memory available for memory pool.
            The actual used memory size is the minimum of the available memory of the device and max_device_memory.
            The format is "xxGB". Default is the maximum available memory of the device, expressed as ``1024GB``.
        optimize_level (str): The memory optimize level. The value must be in ['O0', 'O1']. Default: ``O0`` .

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> ms.set_device("Ascend", 1)
        >>> ms.runtime.set_memory("10GB", "2GB", "60GB", "O1")
    """
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

def _is_initialized(device_target):
    """
    Returns whether specified backend is initialized.
    """
    _device_context = _device_context_mgr.get_device_context(device_target)
    if _device_context is None:
        return False
    return _device_context.initialized()

def memory_stats():
    """
    Returns status information queried from the memory pool.

    Note:
        - For the `CPU` backend, a dictionary with empty data is always returned.

    Returns:
        dict, the queried memory information.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> ms.set_device("Ascend", 0)
        >>> a = Tensor(np.ones([1, 2]), ms.float32)
        >>> b = Tensor(np.ones([1, 2]), ms.float32)
        >>> c = ops.add(a, b).asnumpy()
        >>> print(ms.runtime.memory_stats())
        {'total_reserved_memory': 1073741824, 'total_allocated_memory': 1024, 'total_idle_memory': 1073740800,
        'total_eager_free_memory': 0, 'max_reserved_memory': 1073741824, 'max_allocated_memory': 1536,
        'common_mem_pool_stats': {'block_unit_size': 1073741824, 'block_counts': 1, 'blocks_info':
        {<capsule object NULL at 0x7f7e8c27b030>: {'block_stream_id': 0, 'block_memory_size': 1073741824}}},
        'persistent_mem_pool_stats': {'block_unit_size': 1073741824, 'block_counts': 0, 'blocks_info': {}}}
    """
    device_target = ms.context.get_context("device_target")
    if not _is_initialized(device_target):
        logger.warning(f"Backend {device_target} is not initialized yet. Return empty dict.")
        return {}
    return _memory_stats(device_target)


def memory_reserved():
    """
    Returns the total amount of memory currently managed by the memory pool.

    Note:
        - For the `CPU` backend, 0 is always returned.

    Returns:
        int, in Byte.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> ms.set_device("Ascend", 0)
        >>> a = Tensor(np.ones([1, 2]), ms.float32)
        >>> b = Tensor(np.ones([1, 2]), ms.float32)
        >>> c = ops.add(a, b).asnumpy()
        >>> print(ms.runtime.memory_reserved())
        1073741824
    """
    device_target = ms.context.get_context("device_target")
    if not _is_initialized(device_target):
        logger.warning(f"Backend {device_target} is not initialized yet. Return 0.")
        return 0
    return _memory_stats(device_target).get("total_reserved_memory", 0)


def max_memory_reserved():
    """
    Returns the peak value of the total memory managed by the memory pool since the process was started.

    Note:
        - For the `CPU` backend, 0 is always returned.

    Returns:
        int, in Byte.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> ms.set_device("Ascend", 0)
        >>> a = Tensor(np.ones([1, 2]), ms.float32)
        >>> b = Tensor(np.ones([1, 2]), ms.float32)
        >>> c = ops.add(a, b).asnumpy()
        >>> print(ms.runtime.max_memory_reserved())
        1073741824
    """
    device_target = ms.context.get_context("device_target")
    if not _is_initialized(device_target):
        logger.warning(f"Backend {device_target} is not initialized yet. Return 0.")
        return 0
    return _memory_stats(device_target).get("max_reserved_memory", 0)


def empty_cache():
    """
    Release all memory fragments in the memory pool, so that memory arrangement
    will be optimized.

    Note:
        Currently, the MindSpore memory pool does not have the function of releasing memory fragments.
        This interface is reserved but implemented as an empty method and prompted in log mode.
    """
    logger.warning(f"The empty_cache operation is currently not supported.")


def reset_peak_memory_stats():
    """
    Reset the "peak" stats tracked by memory manager.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> ms.set_device("Ascend", 0)
        >>> a = Tensor(np.ones([1, 2]), ms.float32)
        >>> b = Tensor(np.ones([1, 2]), ms.float32)
        >>> c = ops.add(a, b).asnumpy()
        >>> print(ms.runtime.max_memory_reserved())
        1073741824
        >>> print(ms.runtime.max_memory_allocated())
        1536
        >>> ms.runtime.reset_peak_memory_stats()
        >>> print(ms.runtime.max_memory_reserved())
        0
        >>> print(ms.runtime.max_memory_allocated())
        0
    """
    device_target = ms.context.get_context("device_target")
    _reset_max_mem_reserved(device_target)
    _reset_max_mem_allocated(device_target)


def memory_summary():
    """
    Returns readable memory pool status information.

    Returns:
        str, readable memory pool status information in tabular form.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    device_target = DeviceManagerConf.get_instance().get_device_target()
    stats = _memory_stats(device_target)

    def _format_size(sz, pref_sz):
        prefixes = ["B  ", "KB", "MB", "GB", "TB", "PB"]
        prefix = prefixes[0]
        for new_prefix in prefixes[1:]:
            if pref_sz < 768 * 1024:
                break
            prefix = new_prefix
            sz //= 1024
            pref_sz /= 1024
        return f"{sz:6d} {prefix}"

    metrics_to_display = [
        ("total_reserved_memory", "Reserved memory", _format_size),
        ("total_allocated_memory", "Allocated memory", _format_size),
        ("total_idle_memory", "Idle memory", _format_size),
        ("total_eager_free_memory", "Eager free memory", _format_size),
        ("max_reserved_memory", "Max reserved memory", _format_size),
        ("max_allocated_memory", "Max allocated memory", _format_size),
    ]

    lines = []
    lines.append("=" * 45)
    lines.append(" {:^43} ".format("Memory summary"))
    lines.append("=" * 45)
    lines.append(" {:<20} | {:<20} ".format("Metric", "Data"))

    for metric_key, metric_name, formatter in metrics_to_display:
        lines.append("-" * 45)
        data = stats[metric_key]
        lines.append(" {:<20} | {:<20} ".format(metric_name, formatter(data, data)))

    lines.append("=" * 45)

    return "|" + "|\n|".join(lines) + "|\n"


def memory_allocated():
    """
    Returns the actual memory size currently occupied by Tensor.

    Note:
        - For the `CPU` backend, 0 is always returned.

    Returns:
        int, in Byte.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> ms.set_device("Ascend", 0)
        >>> a = Tensor(np.ones([1, 2]), ms.float32)
        >>> b = Tensor(np.ones([1, 2]), ms.float32)
        >>> c = ops.add(a, b).asnumpy()
        >>> print(ms.runtime.memory_allocated())
        1024
    """
    device_target = ms.context.get_context("device_target")
    if not _is_initialized(device_target):
        logger.warning(f"Backend {device_target} is not initialized yet. Return 0.")
        return 0
    return _memory_stats(device_target).get("total_allocated_memory", 0)


def max_memory_allocated():
    """
    Returns the peak memory size of the memory pool actually occupied by Tensor since the process was started.

    Note:
        - For the `CPU` backend, 0 is always returned.

    Returns:
        int, in Byte.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> ms.set_device("Ascend", 0)
        >>> a = Tensor(np.ones([1, 2]), ms.float32)
        >>> b = Tensor(np.ones([1, 2]), ms.float32)
        >>> c = ops.add(a, b).asnumpy()
        >>> print(ms.runtime.max_memory_allocated())
        1536
    """
    device_target = ms.context.get_context("device_target")
    if not _is_initialized(device_target):
        logger.warning(f"Backend {device_target} is not initialized yet. Return 0.")
        return 0
    return _memory_stats(device_target).get("max_allocated_memory", 0)


def reset_max_memory_reserved():
    """
    Reset the peak memory size managed by the memory pool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> ms.set_device("Ascend", 0)
        >>> a = Tensor(np.ones([1, 2]), ms.float32)
        >>> b = Tensor(np.ones([1, 2]), ms.float32)
        >>> c = ops.add(a, b).asnumpy()
        >>> print(ms.runtime.max_memory_reserved())
        1073741824
        >>> ms.runtime.reset_max_memory_reserved()
        >>> print(ms.runtime.max_memory_reserved())
        0
    """
    device_target = ms.context.get_context("device_target")
    _reset_max_mem_reserved(device_target)


def reset_max_memory_allocated():
    """
    Reset the peak memory size of the memory pool actually occupied by Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> ms.set_device("Ascend", 0)
        >>> a = Tensor(np.ones([1, 2]), ms.float32)
        >>> b = Tensor(np.ones([1, 2]), ms.float32)
        >>> c = ops.add(a, b).asnumpy()
        >>> print(ms.runtime.max_memory_allocated())
        1536
        >>> ms.runtime.reset_max_memory_allocated()
        >>> print(ms.runtime.max_memory_allocated())
        0
    """
    device_target = ms.context.get_context("device_target")
    _reset_max_mem_allocated(device_target)
