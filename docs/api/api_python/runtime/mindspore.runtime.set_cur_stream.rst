mindspore.runtime.set_cur_stream
=================================

.. py:function:: mindspore.runtime.set_cur_stream(stream)

    设置当前流，这是用于设置流的包装器API。

    不建议使用此函数，建议使用 `StreamCtx` 上下文管理器。

    参数：
        - **stream** (Stream) - 指定的流。如果是 ``None`` ，表示此上下文管理器无操作。

    异常：
        - **TypeError** - 参数 `stream` 既不是 :class:`mindspore.runtime.Stream` 也不是 ``None``。
