mindspore.hal.set_cur_stream
=============================

.. py:function:: mindspore.hal.set_cur_stream(stream)

    设置当前流，此接口将在后续版本中废弃，请使用接口 :func:`mindspore.runtime.set_cur_stream` 代替。

    参数：
        - **stream** (Stream) - 指定的流。如果是 ``None`` ，这个上下文管理器无操作。
