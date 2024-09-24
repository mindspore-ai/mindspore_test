mindspore.numpy.split
=================================

.. py:function:: mindspore.numpy.split(x, indices_or_sections, axis=0)

    将一个Tensor沿指定轴分割为多个sub-tensor。

    参数：
        - **x** (Tensor) - 待分割的Tensor。
        - **indices_or_sections** (Union[int, tuple(int), list(int)]) - 如果是整数 :math:`N` ，Tensor将沿轴分割为 :math:`N` 个相等的sub-tensor。如果是tuple(int)、list(int)或排序后的整数，则指示沿轴的分割位置。例如，对于 :math:`axis=0` ， :math:`[2,3]` 将产生三个sub-tensor： :math:`x[:2]` 、 :math:`x[2:3]` 和 :math:`x[3:]` 。如果索引超出轴上数组的维度，则相应地返回空子数组。
        - **axis** (int，可选) - 指定进行分割的轴。默认值： ``0`` 。

    返回：
        sub-tensor的tuple。

    异常：
        - **TypeError** - 如果参数 ``indices_or_sections`` 不是整数、tuple(int)或list(int)，或者参数 ``axis`` 不是整数。
        - **ValueError** - 如果参数 ``axis`` 超出范围 :math:`[-x.ndim, x.ndim)` 。