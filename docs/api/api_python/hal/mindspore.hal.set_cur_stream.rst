mindspore.hal.set_cur_stream
=============================

.. py:function:: mindspore.hal.set_cur_stream(stream)

    设置当前流，这是一个用于设置流的包装器API。

    建议优先使用 `StreamCtx` 上下文管理器，而不是直接使用此函数。

    .. note::
        - 接口即将废弃，请使用接口 :func:`mindspore.runtime.set_cur_stream` 代替。 

    参数：
        - **stream** (Stream) - 指定的流。如果是 ``None`` ，这个上下文管理器无操作。

    异常：
        - **TypeError** - 参数 `stream` 即不是一个 :class:`mindspore.hal.Stream` 也不是一个 ``None``。
