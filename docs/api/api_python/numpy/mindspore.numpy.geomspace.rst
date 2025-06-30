mindspore.numpy.geomspace
=================================

.. py:function:: mindspore.numpy.geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0)

    返回在对数刻度（几何级数）上均匀间隔的数字。
    这和 :func:`mindspore.numpy.logspace` 类似，但直接指定了端点。每个输出样本都是前一个样本的常数倍。

    参数：
        - **start** (Union[int, list(int), tuple(int), tensor]) - 序列的起始值。
        - **stop** (Union[int, list(int), tuple(int), tensor]) - 当 ``endpoint`` 为 ``True`` 时，为序列的最终值；当 ``endpoint`` 为 ``False`` 时，在对数空间内的区间上均匀间隔 num + 1 个值，返回除最后一个值外（长度为 num 的序列）的其他值。
        - **num** (int, 可选) - 生成的样本数。默认值 ``50`` 。
        - **endpoint** (bool, 可选) - 如果为 ``True`` ， ``stop`` 是最后一个样本。否则，它不包括在内。默认值： ``True``。
        - **dtype** (Union[mindspore.dtype, str], 可选) - 指定的Tensor ``dtype`` ，可以是 ``np.float32`` 或 ``float32``。如果 ``dtype`` 为 ``None`` ，则将从其他输入参数推断出数据类型。默认值： ``None`` 。
        - **axis** (int, 可选) - 结果中用于存储样本的轴。仅当 ``start`` 或 ``stop`` 为类似数组对象时才用到。默认值： ``0`` ，默认情况下的采样将沿着在开始处插入的新轴。使用 ``-1`` 在末尾获取一个轴。

    返回：
        Tensor，对数刻度上均匀间隔的样本。

    异常：
        - **TypeError** - 如果输入参数非给定的数据类型。