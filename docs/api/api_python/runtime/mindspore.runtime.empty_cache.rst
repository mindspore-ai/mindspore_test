mindspore.runtime.empty_cache
=============================

.. py:function:: mindspore.runtime.empty_cache()

    清理内存池中的内存碎片，优化内存排布。

    .. note::
        - 目前MindSpore内存池没有清空内存碎片的功能，此为预留接口，实现为空方法。使用时，会通过日志方式提示。
