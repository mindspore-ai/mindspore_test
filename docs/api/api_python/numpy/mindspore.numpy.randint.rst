mindspore.numpy.randint
=================================

.. py:function:: mindspore.numpy.randint(minval, maxval=None, shape=None, dtype=mstype.int32)

    返回从 ``minval`` （包括）到 ``maxval`` （不包括）的随机整数。从“半开”区间范围 :math:`[minval,maxval)` 内的指定数据类型的离散均匀分布中返回随机整数。如果 ``maxval`` 为 ``None`` （默认值），则取值范围为 :math:`[0, minval)` ，此时 ``minval`` 必须大于0。

    参数：
        - **minval** (Union[int]) - 间隔的起始值（包括此值）。当 ``maxval`` 为 ``None`` 时， ``minval`` 必须大于0。当 ``maxval`` 不为 ``None`` 时， ``minval`` 必须小于 ``maxval`` 。
        - **maxval** (Union[int], 可选) - 间隔的结束值（不包括此值）。
        - **shape** (Union[int, tuple(int)]) - 指定的新Tensor的shape，例如 :math:`(2,3)` 或 :math:`2` 。
        - **dtype** (Union[mindspore.dtype, str], 可选) - 指定的Tensor数据类型，必须是int型数据。默认值： ``mstype.int32`` 。

    返回：
        Tensor，给定shape和类型，其中所有元素都为 ``minval`` （包括）到 ``maxval`` （不包括）之间的随机整数。

    异常：
        - **TypeError** - 如果输入参数非上述给定的类型。
        - **ValueError** - 如果输入参数的值不符合上述规定。
