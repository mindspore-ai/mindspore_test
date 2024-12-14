mindspore.runtime
==================

Runtime encapsulates interfaces for stream, event, memory, and executor. MindSpore abstracts the corresponding modules from different backends, allowing users to schedule hardware resources at the Python layer.

Stream
---------

.. msplatformautosummary::
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

Event
---------

.. msplatformautosummary::
    :toctree: runtime
    :nosignatures:
    :template: classtemplate.rst

    mindspore.runtime.Event

Memory
------------

.. msplatformautosummary::
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

Executor
------------

.. msplatformautosummary::
    :toctree: runtime
    :nosignatures:
    :template: classtemplate.rst

    mindspore.runtime.set_cpu_affinity
    mindspore.runtime.launch_blocking
    mindspore.runtime.dispatch_threads_num
