mindspore.numpy.take
=================================

.. py:function:: mindspore.numpy.take(a, indices, axis=None, mode='clip')

    从数组中沿指定轴提取元素。
    当 ``axis`` 不为None时，此函数的功能类似于“fancy”索引（使用数组进行索引）；但如果需要沿给定轴提取元素，它可能更易于使用。例如， ``np.take(arr, indices, axis=3)`` 等价于 ``arr[:,:,:,indices,...]`` 。

    .. note::
        不支持Numpy的 ``out`` 参数。不支持 ``mode = 'raise'`` ，默认 ``mode`` 为 ``'clip'`` 。

    参数：
        - **a** (Tensor) - 源数组，shape为 ``(Ni…, M, Nk…)`` 。
        - **indices** (Tensor) - shape为 ``(Nj...)`` 的索引，表示要提取的值。
        - **axis** (int, 可选) - 用于选择值的轴。默认情况下，使用展平后的输入数组。默认值： ``None`` 。
        - **mode** ('raise', 'wrap', 'clip', 可选) - 指定超出范围的索引的行为。默认值： ``'clip'`` 。

          - 'raise' - 抛出错误；
          - 'wrap' - 循环；
          - 'clip' - 限制在范围内。'clip'模式意味着所有过大的索引都将被替换为该轴最后一个元素的索引。请注意，这将禁止使用负数进行索引。

    返回：
        Tensor，索引结果。

    异常：
        - **ValueError** - 如果 ``axis`` 超出范围。
        - **TypeError** - 如果输入不是Tensor。