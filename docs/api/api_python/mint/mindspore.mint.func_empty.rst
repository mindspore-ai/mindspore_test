mindspore.mint.empty
====================

.. py:function:: mindspore.mint.empty(size, *, dtype=None, device=None)

    创建一个数据没有初始化的Tensor。参数 `size` 指定Tensor的shape，参数 `dtype` 指定填充值的数据类型，参数 `device` 指\
    定Tensor使用的内存来源。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **size** (Union[tuple[int], list[int], int]) - 指定输出Tensor的shape，只允许正整数或者包含正整数\
          的tupled、list。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`, 可选) - 用来描述所创建的Tensor的dtype。如果为 ``None`` ，那\
          么将会使用 `mindspore.float32` 。默认值： ``None`` 。
        - **device** (string, 可选) - 指定Tensor使用的内存来源。当前支持 ``CPU`` 和 ``Ascend`` 。如果为 ``None`` ，
          那么将会使用 :func:`mindspore.set_device` 设置的值。默认值 ``None`` 。

    返回：
        Tensor，shape和dtype由输入定义。

    异常：
        - **TypeError** - 如果 `size` 不是一个int，或元素为int的元组、列表。
