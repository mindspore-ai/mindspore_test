mindspore.Tensor.narrow
=======================

.. py:method:: mindspore.Tensor.narrow(dim, start, length)

    沿着指定的轴，指定起始位置获取指定长度的Tensor。

    参数：
        - **dim** (int) - 指定的轴。
        - **start** (int) - 指定起始位置。
        - **length** (int) - 指定长度。

    返回：
        output (Tensor) - narrow后的Tensor。

    异常：
        - **ValueError** - `dim` 值超出范围[-input.ndim, input.ndim)。
        - **ValueError** - `start` 值超出范围[-input.shape[dim], input.shape[dim]]。
        - **ValueError** - `length` 值超出范围[0, input.shape[dim]-start]。
