mindspore.Tensor.copy\_
=======================

.. py:method:: mindspore.Tensor.copy_(src, non_blocking=False) -> Tensor

    将 `src` 中的元素复制到Tensor中并返回这个Tensor。

    .. warning::
        如果复制在Ascend和Ascend之间进行， `src` 需可广播为 `self`，且 `src` 可以是不同的数据类型。
        CPU和Ascend，CPU和CPU之间的复制仅当 `self` 和 `src` 具有相同的形状和数据类型并且连续时才能支持。

    参数：
        - **src** (Tensor) - 用于复制的Tensor。
        - **non_blocking** (bool，可选) - 如果为 ``True`` ，且复制操作在 CPU 和 Ascend 之间进行，同时 `self` 和 `src` 具有相同的形状和数据类型且连续，则复制操作可能相对于主机异步进行。对于其他情况，此参数无效。默认值为 ``False``。

    返回：
        Tensor，赋值后的Tensor。