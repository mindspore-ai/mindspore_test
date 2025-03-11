mindspore.ops.reshape
======================

.. py:function:: mindspore.ops.reshape(input, shape)

    按指定 `shape` ，对输入tensor进行重排。

    .. note::
        参数 `shape` 中的-1表示该维度值是从其他维度和输入的元素数量推断出来的。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **shape** (Union[tuple[int], list[int], Tensor[int]]) - 新shape。

    返回：
        Tensor
