mindspore.numpy.nanstd
======================

.. py:function:: mindspore.numpy.nanstd(a, axis=None, dtype=None, ddof=0, keepdims=False)

    计算指定轴上元素的标准差，忽略NaN。

    返回非NaN数组元素的标准差，即分布的离散程度。默认情况下，标准差在展平数组上计算，否则在指定的轴上计算。

    .. note::
        不支持NumPy参数 `out` 。在GPU上，支持的数据类型为np.float16和np.float32。

    参数：
        - **a** (Union[int, float, list, tuple, Tensor]) - 计算非NaN值的标准差。
        - **axis** (Union[int, tuple of int, None], 可选) - 计算标准差所沿的单个或多个轴。 若取默认值，计算展平数组的标准差。默认值: `None` 。
        - **dtype** (mindspore.dtype, 可选) - 默认值: `None` 。 覆盖输出Tensor的dtype。
        - **ddof** (int, 可选) - “自由度偏差”: 计算中使用的除数是 ``N - ddof`` ，其中 `N` 表示非NaN元素的数量。默认情况下 `ddof` 为零。
        - **keepdims** (boolean, 可选) - 默认值:  `False` 。如果设置为 `True` ，减少的轴在结果中保留为大小为1的维度。 若使用此选项，结果会广播到和 `a` 同一个维度数。

    返回：
        Tensor。
    
    异常：
        - **ValueError** - 如果 `axis` 超出[-a.ndim, a.ndim)范围，或者 `axis` 包含重复项。