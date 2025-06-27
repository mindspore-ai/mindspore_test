mindspore.runtime
==================

Runtime encapsulates interfaces for executor, memory, stream and event. MindSpore abstracts the corresponding modules from different backends, allowing users to schedule hardware resources at the Python layer.

Executor
------------

.. autosummary::
    :toctree: runtime
    :nosignatures:
    :template: classtemplate.rst

    mindspore.runtime.dispatch_threads_num
    mindspore.runtime.launch_blocking
    mindspore.runtime.set_cpu_affinity
    mindspore.runtime.set_kernel_launch_capture
    mindspore.runtime.set_kernel_launch_group

Memory
------------

.. autosummary::
    :toctree: runtime
    :nosignatures:
    :template: classtemplate.rst

    mindspore.runtime.max_memory_allocated
    mindspore.runtime.max_memory_reserved
    mindspore.runtime.memory_allocated
    mindspore.runtime.memory_replay
    mindspore.runtime.memory_reserved
    mindspore.runtime.memory_stats
    mindspore.runtime.memory_summary
    mindspore.runtime.reset_max_memory_reserved
    mindspore.runtime.reset_max_memory_allocated
    mindspore.runtime.reset_peak_memory_stats
    mindspore.runtime.empty_cache
    mindspore.runtime.set_memory

Stream
---------

.. autosummary::
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

.. autosummary::
    :toctree: runtime
    :nosignatures:
    :template: classtemplate.rst

    mindspore.runtime.Event
