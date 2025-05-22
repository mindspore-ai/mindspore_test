mindspore.Tensor.storage
========================

.. py:method:: mindspore.Tensor.storage()

    返回Tensor的存储，该存储不区分dtype，当前只支持Ascend平台。


    返回：
        UntypedStorage，底层的Storage实现。

    异常：
        - **RuntimeError** - Tensor的存储不在Ascend上。
