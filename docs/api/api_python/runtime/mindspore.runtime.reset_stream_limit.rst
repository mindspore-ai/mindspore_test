mindspore.runtime.reset_stream_limit
==================================

.. py:function:: mindspore.runtime.reset_stream_limit(stream)

    复位指定流上限制核数。

    .. note::
        该接口会同步下发和执行，可能会影响性能。

    参数：
        - **stream** (:class:`mindspore.runtime.Stream`) - 指定的流对象。
