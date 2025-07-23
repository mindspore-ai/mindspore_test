mindspore.runtime.set_kernel_launch_capture
=============================================

.. py:function:: mindspore.runtime.set_kernel_launch_capture(enable_capture_graph, op_capture_skip=None)

    O0/O1 模式下，增量推理场景支持捕获计算图。通过将CPU侧算子调度行为捕获为一个计算图，可以提高CPU侧算子调度的性能。

    .. warning::
        这是一个实验性的接口，未来可能会被更改或删除。

    参数：
        - **enable_capture_graph** (bool) - 是否启用计算图捕获。它可以在脚本中的任意位置开启或关闭。
        - **op_capture_skip** (list) - 自定义不捕获的算子名，默认为 ``None``。
