mindspore.ops.argsort
======================

.. py:function:: mindspore.ops.argsort(input, axis=-1, descending=False)

    返回按指定轴对tensor进行排序后的索引。

    .. note::
        当前Ascend后端只支持对最后一维进行排序。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **axis** (int) - 指定轴。默认 ``-1`` 。
        - **descending** (bool) - 指定排序（升序或降序）。默认 ``False`` 。

    返回：
        Tensor
