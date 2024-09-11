mindspore.hal
=============

Hal中封装了设备管理、流管理、事件管理与内存管理的接口。MindSpore从不同后端抽象出对应的上述模块，允许用户在Python层调度硬件资源。

设备管理
------------

.. mscnplatformautosummary::
    :toctree: hal
    :nosignatures:
    :template: classtemplate.rst

    mindspore.hal.device_count
    mindspore.hal.get_arch_list
    mindspore.hal.get_device_capability
    mindspore.hal.get_device_name
    mindspore.hal.get_device_properties
    mindspore.hal.is_available
    mindspore.hal.is_initialized

流管理
------------

.. mscnplatformautosummary::
    :toctree: hal
    :nosignatures:
    :template: classtemplate.rst

    mindspore.hal.current_stream
    mindspore.hal.default_stream
    mindspore.hal.set_cur_stream
    mindspore.hal.synchronize
    mindspore.hal.Stream
    mindspore.hal.StreamCtx

事件管理
------------

.. mscnplatformautosummary::
    :toctree: hal
    :nosignatures:
    :template: classtemplate.rst

    mindspore.hal.Event
    mindspore.hal.CommHandle

内存管理
------------

.. mscnplatformautosummary::
    :toctree: hal
    :nosignatures:
    :template: classtemplate.rst

    mindspore.hal.contiguous_tensors_handle.combine_tensor_list_contiguous
    mindspore.hal.contiguous_tensors_handle.ContiguousTensorsHandle
    mindspore.hal.max_memory_allocated
    mindspore.hal.max_memory_reserved
    mindspore.hal.memory_allocated
    mindspore.hal.memory_reserved
    mindspore.hal.memory_stats
    mindspore.hal.memory_summary
    mindspore.hal.reset_max_memory_reserved
    mindspore.hal.reset_max_memory_allocated
    mindspore.hal.reset_peak_memory_stats
    mindspore.hal.empty_cache
