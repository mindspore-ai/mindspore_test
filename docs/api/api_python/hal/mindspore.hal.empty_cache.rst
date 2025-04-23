mindspore.hal.empty_cache
=========================

.. py:function:: mindspore.hal.empty_cache()

    清理内存池中的内存碎片，优化内存排布，此接口将在后续版本中废弃，请使用接口 :func:`mindspore.runtime.empty_cache` 代替。

    .. note::
        - 清空缓存可能有助于减少设备内存碎片，但可能对网络性能产生负面影响。
        - 不支持Atlas训练系列产品。
