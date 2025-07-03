mindspore.numpy.linspace
=================================

.. py:function:: mindspore.numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)

    返回给定区间内均匀间隔的值。

    参数：
        - **start** (Union[int, list(int), tuple(int), tensor]) - 序列的起始值。
        - **stop** (Union[int, list(int), tuple(int), tensor]) - 当 ``endpoint`` 为 ``True`` 时，为序列的最终值；当 ``endpoint`` 为 ``False`` 时，在给定区间上均匀间隔 num + 1 个值，返回除最后一个值外（长度为 num 的序列）的其他值。
        - **num** (int, 可选) - 要生成的等间隔样例数量，默认值： ``50`` 。
        - **endpoint** (bool, 可选) - 序列中是否包含 ``stop`` 值，默认值： ``True`` 。
        - **retstep** (bool, 可选) - 如果为 ``True`` ，则返回包含样例之间的 ``step``  ``(samples, step)`` ，其中 ``step`` 是生成数值的间隔。
        - **dtype** (Union[mindspore.dtype, str], 可选) - 指定的Tensor ``dtype`` 。如果 ``dtype`` 为 ``None`` ，则将从其他输入参数推断出新Tensor的数据类型。默认值： ``None`` 。
        - **axis** (int, 可选) - 结果中用于存储样本的轴。仅当 ``start`` 或 ``stop`` 为类似数组对象时才用到。默认值： ``0`` ，默认情况下的采样将沿着在开始处插入的新轴。使用 ``-1`` 在末尾获取一个轴。

    返回：
        Tensor，给定区间范围内以均匀间隔生成 ``num`` 个值。如果 ``endpoint`` 为 ``True`` ，数值范围为： :math:`[start,stop]` ；反之则为：:math:`[start,stop)` 。
        Step，样本之间间距的大小，仅当 ``retstep`` 为 ``True`` 时返回。

    异常：
        - **TypeError** - 如果输入参数为非给定的数据类型。