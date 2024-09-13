mindspore.numpy.select
=================================

.. py:function:: mindspore.numpy.select(condlist, choicelist, default=0)

    根据条件从 ``choicelist`` 中的元素中返回数组。

    参数：
        - **condlist** (Union[int, float, bool, list, tuple, Tensor]) - 条件列表，用于确定从 ``choicelist`` 中哪个数组中取出输出元素。当多个条件满足时，使用 ``condlist`` 中遇到的第一个条件。
        - **choicelist** (Union[int, float, bool, list, tuple, Tensor]) - 数组列表，从中取出输出元素。其长度必须与 ``condlist`` 相同。
        - **default** (scalar, 可选) - 当所有条件均为 ``False`` 时插入输出的元素。默认值: ``0`` 。

    返回：
        Tensor，输出位置 ``m`` 是 ``choicelist`` 中数组的第 ``m-th`` 个元素，其中 ``condlist`` 对应数组的第 ``m-th`` 个元素为 ``True`` 。

    异常：
        - **ValueError** - 如果 :math:`len(condlist) != len(choicelist)` 。