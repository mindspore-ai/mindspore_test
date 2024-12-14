mindspore.runtime
==================

运行时封装了流、事件、内存、执行的接口。MindSpore从不同的后端抽象出相应的模块，允许用户在Python层调度硬件资源。

流
---------

.. mscnautosummary::
    :toctree: runtime
    :nosignatures:
    :template: classtemplate.rst

    mindspore.runtime.communication_stream
    mindspore.runtime.current_stream
    mindspore.runtime.default_stream
    mindspore.runtime.set_cur_stream
    mindspore.runtime.synchronize
    mindspore.runtime.Stream
    mindspore.runtime.StreamCtx

事件
---------

.. mscnautosummary::
    :toctree: runtime
    :nosignatures:
    :template: classtemplate.rst

    mindspore.runtime.Event

内存
------------

.. mscnautosummary::
    :toctree: runtime
    :nosignatures:
    :template: classtemplate.rst

    mindspore.runtime.max_memory_allocated
    mindspore.runtime.max_memory_reserved
    mindspore.runtime.memory_allocated
    mindspore.runtime.memory_reserved
    mindspore.runtime.memory_stats
    mindspore.runtime.memory_summary
    mindspore.runtime.reset_max_memory_reserved
    mindspore.runtime.reset_max_memory_allocated
    mindspore.runtime.reset_peak_memory_stats
    mindspore.runtime.empty_cache
    mindspore.runtime.set_memory

执行
------------

.. msplatformautosummary::
    :toctree: runtime
    :nosignatures:
    :template: classtemplate.rst

    mindspore.runtime.set_cpu_affinity
    mindspore.runtime.launch_blocking
    mindspore.runtime.dispatch_threads_num
