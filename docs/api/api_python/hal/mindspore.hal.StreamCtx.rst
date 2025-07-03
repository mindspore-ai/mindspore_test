mindspore.hal.StreamCtx
==========================

.. py:class:: mindspore.hal.StreamCtx(ctx_stream)

    上下文管理器，用于选择给定的流，此接口将在后续版本中废弃，请使用接口 :class:`mindspore.runtime.StreamCtx` 代替。

    在上下文范围内，所有算子都将在指定流上执行。

    参数：
        - **ctx_stream** (Stream) - 指定的流。如果是 `None` ，则该上下文管理器无操作。
