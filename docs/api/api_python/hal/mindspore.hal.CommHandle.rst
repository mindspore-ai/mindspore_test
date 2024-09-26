mindspore.hal.CommHandle
========================

.. py:class:: mindspore.hal.CommHandle()

    通常CommHandle是在C++中执行通信算子时创建并返回给Python
    层，它不会直接在Python中创建。只有在GRAPH模式下，才会在Python创建CommHandles。

    .. py:method:: wait()

        如果CommHandle是在Python创建的，那么对其调用wait不会生效。
