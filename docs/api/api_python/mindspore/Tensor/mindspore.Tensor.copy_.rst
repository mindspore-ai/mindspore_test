mindspore.Tensor.copy\_
=======================

.. py:method:: mindspore.Tensor.copy_(src, non_blocking=False)

    将src中的元素复制到Tensor中并返回这个Tensor。

    .. warning::
        这是一个实验性API，后续可能修改或删除。
        `src` 需可广播为 `self`。 `src` 可以是不同的数据类型。

    参数：
        - **src** (Tensor) - 用于复制的Tensor。
        - **non_blocking** (bool) - 当前无作用。

    返回：
        Tensor，赋值后的Tensor。