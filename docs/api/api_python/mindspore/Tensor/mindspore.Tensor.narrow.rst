mindspore.Tensor.narrow
=======================

.. py:method:: mindspore.Tensor.narrow(dim, start, length) -> Tensor

    沿着指定的轴，从指定的起始位置获取指定长度的Tensor。

    参数：
        - **dim** (int) - 指定的轴。
        - **start** (Union[int, Tensor]) - 指定的起始位置。
        - **length** (int) - 指定长度。

    返回：
        output (Tensor) - narrow后的Tensor。

    异常：
        - **ValueError** - `dim` 值超出范围[-self.ndim, self.ndim)。
        - **ValueError** - `start` 值超出范围[-self.shape[dim], self.shape[dim]]。
        - **ValueError** - `length` 值超出范围[0, self.shape[dim]-start]。
