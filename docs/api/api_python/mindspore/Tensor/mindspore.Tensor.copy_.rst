mindspore.Tensor.copy_
======================

.. py:method:: mindspore.Tensor.copy_(src, non_blocking=False)

    将另一个Tensor的值赋给当前Tensor,

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    .. note::
        当前实现不支持类似Torch的 `non_blocking` 参数。

    参数：
        - **src** (Tensor) - 用于复制的Tensor。
        - **non_blocking** (bool) - 如果为 ``True``，并且复制是在CPU和NPU之间进行的，则复制后的拷贝可能会与源信息异步。对于其他类型的复制操作则该参数不会发生作用。

    返回：
        Tensor，赋值后的Tensor。