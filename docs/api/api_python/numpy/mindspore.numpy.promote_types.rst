mindspore.numpy.promote_types
=============================

.. py:function:: mindspore.numpy.promote_types(type1, type2)

    返回 `type1` 与 `type2` 都可以安全转换的最小位数和最小标量类型的数据类型。

    .. note::
        类型提升规则与原版NumPy略有不同，更类似于jax，因为其更倾向于32位而非64位数据类型。

    参数：
        - **type1** (Union[mindspore.dtype, str]) - 第一个数据类型。
        - **type2** (Union[mindspore.dtype, str]) - 第二个数据类型。

    返回：
        提升后的数据类型。

    异常：
        - **TypeError** - 如果输入不是有效的mindspore.dtype输入。